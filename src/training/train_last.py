import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm
import gc
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
import torch.nn.functional as F

from src.clip.model import convert_state_dict
import matplotlib.pyplot as plt
import numpy as np

def all_gather_with_grad(tensor):
    """实现一个支持梯度传播的 all_gather 操作"""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # 执行常规 all_gather 操作
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # 创建自定义函数来反向传播梯度
    class AllGatherWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, rank):
            ctx.rank = rank
            return torch.cat(tensor_list, dim=0)
        
        @staticmethod
        def backward(ctx, grad_output):
            # 只对应于当前进程的部分需要梯度
            grad_input = torch.zeros_like(grad_output.chunk(world_size)[ctx.rank])
            
            # 提取当前进程对应的梯度部分
            grad_slice = grad_output.chunk(world_size)[ctx.rank]
            grad_input.copy_(grad_slice)
            
            return grad_input, None
    
    # 应用自定义函数
    return AllGatherWithGrad.apply(tensor, rank)
    
class LossLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.start_step = None  # 新增：记录起始step
        
        # Training metrics
        self.train_total_losses = []        
        self.train_ppm_losses = []          
        self.train_pdc_losses = []          
        self.train_plc_losses = []  
        self.train_accs = []                
        self.train_steps = []
        
        # Validation metrics
        self.val_total_losses = []
        self.val_ppm_losses = []
        self.val_pdc_losses = []
        self.val_plc_losses = []
        self.val_top1_accs = []
        self.val_top5_accs = []
        self.val_top10_accs = []
        self.val_steps = []
        
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            
    def _normalize_step(self, step):
        """新增：将step归一化为从0开始"""
        if self.start_step is None:
            self.start_step = step
        return step - self.start_step
    
    def save_history(self):
        """保存训练历史到文件"""
        if not self.log_dir:
            return
            
        history = {
            'start_step': self.start_step,  # 新增：保存起始步数
            'train': {
                'total_losses': self.train_total_losses,
                'ppm_losses': self.train_ppm_losses,
                'pdc_losses': self.train_pdc_losses,
                'plc_losses': self.train_plc_losses,
                'accs': self.train_accs,
                'steps': self.train_steps
            },
            'val': {
                'total_losses': self.val_total_losses,
                'ppm_losses': self.val_ppm_losses,
                'pdc_losses': self.val_pdc_losses,
                'plc_losses': self.val_plc_losses,
                'top1_accs': self.val_top1_accs,
                'top5_accs': self.val_top5_accs,
                'top10_accs': self.val_top10_accs,
                'steps': self.val_steps
            }
        }
        
        np.save(os.path.join(self.log_dir, 'training_history.npy'), history)
        
    def load_history(self):
        """从文件加载训练历史"""
        history_path = os.path.join(self.log_dir, 'training_history.npy')
        if not os.path.exists(history_path):
            return False
            
        try:
            history = np.load(history_path, allow_pickle=True).item()
            self.start_step = history.get('start_step', None)
            # 恢复训练数据
            self.train_total_losses = history['train']['total_losses']
            self.train_ppm_losses = history['train']['ppm_losses']
            self.train_pdc_losses = history['train']['pdc_losses']
            self.train_plc_losses = history['train']['plc_losses']
            self.train_accs = history['train']['accs']
            self.train_steps = history['train']['steps']
            
            # 恢复验证数据
            self.val_total_losses = history['val']['total_losses']
            self.val_ppm_losses = history['val']['ppm_losses']
            self.val_pdc_losses = history['val']['pdc_losses']
            self.val_plc_losses = history['val']['plc_losses']
            self.val_top1_accs = history['val']['top1_accs']
            self.val_top5_accs = history['val']['top5_accs']
            self.val_top10_accs = history['val']['top10_accs']
            self.val_steps = history['val']['steps']
            
            return True
        except Exception as e:
            logging.warning(f"Failed to load training history: {str(e)}")
            return False
    
    def log_train(self, total_loss, ppm_loss, pdc_loss, plc_loss,  acc, step):
        """记录训练指标"""
        if None not in (total_loss, ppm_loss, pdc_loss, plc_loss, step):
            normalized_step = self._normalize_step(step)  # 修改：使用归一化的step
            self.train_total_losses.append(float(total_loss))
            self.train_ppm_losses.append(float(ppm_loss))
            self.train_pdc_losses.append(float(pdc_loss))
            self.train_plc_losses.append(float(plc_loss))
            self.train_steps.append(int(normalized_step))
            if acc is not None:
                self.train_accs.append(float(acc))
            
    def log_val(self, total_loss, ppm_loss, pdc_loss, plc_loss, 
            top1_acc, top5_acc, top10_acc, step):
        """记录验证指标"""
        if None not in (total_loss, ppm_loss, pdc_loss, plc_loss,
                       top1_acc, top5_acc, top10_acc, step):
            normalized_step = self._normalize_step(step)  # 修改：使用归一化的step
            self.val_total_losses.append(float(total_loss))
            self.val_ppm_losses.append(float(ppm_loss))
            self.val_pdc_losses.append(float(pdc_loss))
            self.val_plc_losses.append(float(plc_loss))
            self.val_top1_accs.append(float(top1_acc))
            self.val_top5_accs.append(float(top5_acc))
            self.val_top10_accs.append(float(top10_acc))
            self.val_steps.append(int(normalized_step))
    
    def plot_and_save(self):
        """绘制并保存图表和数据"""
        if not self.log_dir:
            return
            
        try:
            # 1. Training Loss Curves
            if self.train_total_losses:
                plt.figure(figsize=(10, 6))
                plt.plot(self.train_steps, self.train_total_losses, label='Total Loss')
                plt.plot(self.train_steps, self.train_ppm_losses, label='PPM Loss')
                plt.plot(self.train_steps, self.train_pdc_losses, label='PDC Loss')
                plt.plot(self.train_steps, self.train_plc_losses, label='PLC Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Training Losses')
                plt.legend()
                plt.grid(True)
                plt.xlim(left=0)  # 修改：确保x轴从0开始
                plt.savefig(os.path.join(self.log_dir, 'train_loss_curve.png'))
                plt.close()
    
            # 2. Validation Loss Curves
            if self.val_total_losses:
                plt.figure(figsize=(10, 6))
                plt.plot(self.val_steps, self.val_total_losses, label='Total Loss')
                plt.plot(self.val_steps, self.val_ppm_losses, label='PPM Loss')
                plt.plot(self.val_steps, self.val_pdc_losses, label='PDC Loss')
                plt.plot(self.val_steps, self.val_plc_losses, label='PLC Loss')
                plt.xlabel('Steps')
                plt.ylabel('Loss')
                plt.title('Validation Losses')
                plt.legend()
                plt.grid(True)
                plt.xlim(left=0)  # 修改：确保x轴从0开始
                plt.savefig(os.path.join(self.log_dir, 'val_loss_curve.png'))
                plt.close()
    
            # 3. Training Accuracy Curve
            if self.train_accs:
                plt.figure(figsize=(10, 6))
                plt.plot(self.train_steps, self.train_accs, label='Training Accuracy')
                plt.xlabel('Steps')
                plt.ylabel('Accuracy (%)')
                plt.title('Training Accuracy')
                plt.legend()
                plt.grid(True)
                plt.xlim(left=0)  # 修改：确保x轴从0开始
                plt.savefig(os.path.join(self.log_dir, 'train_accuracy_curve.png'))
                plt.close()
    
            # 4. Validation Accuracy Curves
            if self.val_top1_accs:
                plt.figure(figsize=(10, 6))
                plt.plot(self.val_steps, self.val_top1_accs, label='Top1 Acc')
                plt.plot(self.val_steps, self.val_top5_accs, label='Top5 Acc')
                plt.plot(self.val_steps, self.val_top10_accs, label='Top10 Acc')
                plt.xlabel('Steps')
                plt.ylabel('Accuracy (%)')
                plt.title('Validation Accuracy')
                plt.legend()
                plt.grid(True)
                plt.xlim(left=0)  # 修改：确保x轴从0开始
                plt.savefig(os.path.join(self.log_dir, 'val_accuracy_curve.png'))
                plt.close()
    
            # Save normalized data
            if self.train_total_losses:
                np.save(os.path.join(self.log_dir, 'train_total_losses.npy'), np.array(self.train_total_losses))
                np.save(os.path.join(self.log_dir, 'train_ppm_losses.npy'), np.array(self.train_ppm_losses))
                np.save(os.path.join(self.log_dir, 'train_pdc_losses.npy'), np.array(self.train_pdc_losses))
                np.save(os.path.join(self.log_dir, 'train_plc_losses.npy'), np.array(self.train_plc_losses))
                np.save(os.path.join(self.log_dir, 'train_steps.npy'), np.array(self.train_steps))
                if self.train_accs:
                    np.save(os.path.join(self.log_dir, 'train_accs.npy'), np.array(self.train_accs))
    
            if self.val_total_losses:
                np.save(os.path.join(self.log_dir, 'val_total_losses.npy'), np.array(self.val_total_losses))
                np.save(os.path.join(self.log_dir, 'val_ppm_losses.npy'), np.array(self.val_ppm_losses))
                np.save(os.path.join(self.log_dir, 'val_pdc_losses.npy'), np.array(self.val_pdc_losses))
                np.save(os.path.join(self.log_dir, 'val_plc_losses.npy'), np.array(self.val_plc_losses))
                np.save(os.path.join(self.log_dir, 'val_steps.npy'), np.array(self.val_steps))
                np.save(os.path.join(self.log_dir, 'val_top1_accs.npy'), np.array(self.val_top1_accs))
                np.save(os.path.join(self.log_dir, 'val_top5_accs.npy'), np.array(self.val_top5_accs))
                np.save(os.path.join(self.log_dir, 'val_top10_accs.npy'), np.array(self.val_top10_accs))
                
        except Exception as e:
            logging.error(f"Error in plot_and_save: {str(e)}", exc_info=True)

def is_master(args):
    return args.rank == 0

def compute_similarity(x, y, logit_scale):
    """计算两个特征矩阵的相似度"""
    x = x / (x.norm(dim=-1, keepdim=True) + 1e-10)
    y = y / (y.norm(dim=-1, keepdim=True) + 1e-10)
    return logit_scale * (x @ y.T)

def get_ppm_loss(visual_only1, text_only1, fused1, fused2, batch_size, args):
    """
    Product to Product Matching Loss - 论文实现版本
    计算三种相似度矩阵: 多模态-多模态, 视觉-多模态, 文本-多模态
    """
    device = fused1.device
    
    # 计算三种相似度矩阵
    s_m2m = torch.matmul(fused1, fused2.t())  # 多模态-多模态
    s_v2m = torch.matmul(visual_only1, fused2.t())  # 视觉-多模态
    s_t2m = torch.matmul(text_only1, fused2.t())  # 文本-多模态
    
    # 创建标签矩阵 (对角线为1，表示正样本)
    y_matrix = torch.eye(batch_size, device=device)
    
    # 提取对角线元素（正样本相似度）
    diag_indices = torch.arange(batch_size, device=device)
    s_m2m_ii = s_m2m[diag_indices, diag_indices].unsqueeze(1)
    s_v2m_ii = s_v2m[diag_indices, diag_indices].unsqueeze(1)
    s_t2m_ii = s_t2m[diag_indices, diag_indices].unsqueeze(1)
    
    # 计算三部分损失
    loss_m2m = torch.maximum(
        torch.zeros_like(s_m2m),
        args.alpha1 * (1 - y_matrix) + s_m2m - s_m2m_ii
    )
    
    loss_v2m = torch.maximum(
        torch.zeros_like(s_v2m),
        args.alpha1 * (1 - y_matrix) + s_v2m - s_v2m_ii
    )
    
    loss_t2m = torch.maximum(
        torch.zeros_like(s_t2m),
        args.alpha1 * (1 - y_matrix) + s_t2m - s_t2m_ii
    )
    
    # 合并三部分损失
    total_loss = (loss_m2m.sum() + loss_v2m.sum() + loss_t2m.sum()) / (3 * batch_size * batch_size)
    
    return total_loss

def get_pdc_loss(m_sim, v_sim, t_sim, batch_size, args):  # 添加args参数
    """
    Product Self-Distinctiveness Loss
    """
    diag_indices = torch.arange(batch_size, device=m_sim.device)
    v_diag = v_sim[diag_indices, diag_indices].unsqueeze(1)
    t_diag = t_sim[diag_indices, diag_indices].unsqueeze(1)
    
    loss_v = torch.maximum(
        torch.zeros_like(m_sim),
        args.alpha2 + m_sim - v_diag
    )
    
    loss_t = torch.maximum(
        torch.zeros_like(m_sim),
        args.alpha2 + m_sim - t_diag
    )
    
    return (loss_v.sum() + loss_t.sum()) / (batch_size * batch_size)

def get_plc_loss(m_sim, v_sim, t_sim, batch_size, args):  # 添加args参数
    """
    Product Locality Consistency Loss
    """
    loss = 0.0
    k = min(10, batch_size)
    _, top_indices = torch.topk(m_sim, k=k, dim=1)
    
    for i in range(batch_size):
        indices = top_indices[i]
        m_vals = m_sim[i, indices]
        v_vals = v_sim[i, indices]
        t_vals = t_sim[i, indices]
        
        loss_vm = torch.maximum(
            torch.zeros_like(m_vals),
            -args.alpha3 + (v_vals - m_vals).pow(2)
        )
        
        loss_tm = torch.maximum(
            torch.zeros_like(m_vals),
            -args.alpha3 + (t_vals - m_vals).pow(2)
        )
        
        loss_vt = torch.maximum(
            torch.zeros_like(m_vals),
            -args.alpha3 + (v_vals - t_vals).pow(2)
        )
        
        loss += (loss_vm + loss_tm + loss_vt).sum()
    
    return loss / (3 * batch_size * batch_size)

import torch.nn.functional as F

# 添加新的损失函数
def modal_alignment_loss(visual_feats, text_feats, fusion_feats):
    """特征空间对齐损失，确保不同模态特征在同一向量空间，支持模态缺失"""
    loss = 0.0
    count = 0
    
    # 如果有视觉特征，计算视觉-融合对齐损失
    if visual_feats is not None:
        visual_feats = F.normalize(visual_feats, p=2, dim=1)
        v2f_sim = torch.sum(visual_feats * F.normalize(fusion_feats, p=2, dim=1), dim=1)
        loss += (2.0 - torch.mean(v2f_sim))
        count += 1
    
    # 如果有文本特征，计算文本-融合对齐损失
    if text_feats is not None:
        text_feats = F.normalize(text_feats, p=2, dim=1)
        t2f_sim = torch.sum(text_feats * F.normalize(fusion_feats, p=2, dim=1), dim=1)
        loss += (2.0 - torch.mean(t2f_sim))
        count += 1
    
    # 返回平均损失
    return loss / max(count, 1)


def cross_modal_contrastive_loss(visual_feats, text_feats, temperature=0.07):
    """跨模态对比学习损失，促进图像和文本特征对齐"""
    # 归一化特征
    visual_feats = F.normalize(visual_feats, p=2, dim=1)
    text_feats = F.normalize(text_feats, p=2, dim=1)
    
    # 计算相似度矩阵
    logits = torch.matmul(visual_feats, text_feats.t()) / temperature
    
    # 对角线为正样本
    labels = torch.arange(len(visual_feats), device=visual_feats.device)
    
    # 计算对比损失
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    return (loss_i2t + loss_t2i) / 2
def improved_contrastive_loss(feats1, feats2, temperature=0.07, margin=0.2, hard_negative_ratio=0.5):
    # 归一化特征
    feats1 = F.normalize(feats1, p=2, dim=1)
    feats2 = F.normalize(feats2, p=2, dim=1)
    
    # 计算相似度矩阵并应用温度缩放
    logits = torch.matmul(feats1, feats2.t()) / temperature
    
    # 对角线元素为正样本对
    batch_size = len(feats1)
    labels = torch.arange(batch_size, device=feats1.device)
    
    # 获取正样本对相似度（对角线元素）
    pos_sims = torch.diag(logits)
    
    # 获取负样本对相似度（非对角线元素）
    neg_sims = logits - torch.eye(batch_size, device=logits.device) * 1000.0  # 掩蔽掉正样本
    
    # 困难负样本挖掘 - 只保留最相似的前k%个负样本
    k = int(batch_size * hard_negative_ratio)
    hardest_negatives, _ = torch.topk(neg_sims, k, dim=1)
    
    # 基于间隔的损失，用于困难负样本
    hard_negative_loss = F.relu(margin + hardest_negatives - pos_sims.unsqueeze(1)).mean()
    
    # 标准对比损失
    standard_loss = F.cross_entropy(logits, labels)
    
    # 组合两种损失
    return standard_loss + hard_negative_loss
# 修改后的get_loss函数
def get_loss(model, images1, texts1, images2, texts2, tags, item_ids1, item_ids2, loss_fn, args):
    """主损失函数，符合论文实现，同时保留原有的额外损失函数"""
    # 获取模型输出
    outputs = model(images1, texts1, images2, texts2)
    
    if len(outputs) == 7:
        (visual_only1, text_only1, fused1,
         visual_only2, text_only2, fused2,
         logit_scale) = outputs
    else:
        (visual_only1, text_only1, fused1, logit_scale) = outputs
        return NotImplementedError("只支持双组输入")
    
    batch_size = len(images1)
    
    # 计算相似度矩阵
    sim_m2m = torch.matmul(fused1, fused2.t()) * logit_scale  # 多模态-多模态
    sim_v2m = torch.matmul(visual_only1, fused2.t()) * logit_scale  # 视觉-多模态
    sim_t2m = torch.matmul(text_only1, fused2.t()) * logit_scale  # 文本-多模态
    
    # 额外的相似度矩阵，用于原有损失
    sim_v2v = torch.matmul(visual_only1, visual_only2.t()) * logit_scale  # 视觉-视觉
    sim_t2t = torch.matmul(text_only1, text_only2.t()) * logit_scale  # 文本-文本
    sim_v2t = torch.matmul(visual_only1, text_only2.t()) * logit_scale  # 视觉-文本
    
    # 创建标签矩阵 (对角线为1，表示正样本)
    y_matrix = torch.eye(batch_size, device=sim_m2m.device)
    
    # 提取对角线元素（正样本相似度）
    diag_indices = torch.arange(batch_size, device=sim_m2m.device)
    s_m2m_ii = sim_m2m[diag_indices, diag_indices].unsqueeze(1)
    s_v2m_ii = sim_v2m[diag_indices, diag_indices].unsqueeze(1)
    s_t2m_ii = sim_t2m[diag_indices, diag_indices].unsqueeze(1)
    
    # 1. PPM损失 - 产品到产品匹配损失 (论文方式)
    ppm_m2m_loss = torch.maximum(
        torch.zeros_like(sim_m2m),
        args.alpha1 * (1 - y_matrix) + sim_m2m - s_m2m_ii
    ).sum()
    
    ppm_v2m_loss = torch.maximum(
        torch.zeros_like(sim_v2m),
        args.alpha1 * (1 - y_matrix) + sim_v2m - s_v2m_ii
    ).sum()
    
    ppm_t2m_loss = torch.maximum(
        torch.zeros_like(sim_t2m),
        args.alpha1 * (1 - y_matrix) + sim_t2m - s_t2m_ii
    ).sum()
    
    ppm_loss = (ppm_m2m_loss + ppm_v2m_loss + ppm_t2m_loss) / (3 * batch_size * batch_size)
    
    # 2. PDC损失 - 产品自身区分性损失 (论文方式)
    pdc_v_loss = torch.maximum(
        torch.zeros_like(sim_m2m),
        args.alpha2 + sim_m2m - s_v2m_ii
    ).sum()
    
    pdc_t_loss = torch.maximum(
        torch.zeros_like(sim_m2m),
        args.alpha2 + sim_m2m - s_t2m_ii
    ).sum()
    
    pdc_loss = (pdc_v_loss + pdc_t_loss) / (2 * batch_size * batch_size)
    
    # 3. PLC损失 - 产品局部一致性损失 (论文方式)
    k = min(10, batch_size)
    _, top_indices = torch.topk(sim_m2m, k=k, dim=1)
    
    plc_loss = 0.0
    for i in range(batch_size):
        indices = top_indices[i]
        m_vals = sim_m2m[i, indices]
        v_vals = sim_v2m[i, indices]
        t_vals = sim_t2m[i, indices]
        
        loss_vm = torch.maximum(
            torch.zeros_like(m_vals),
            -args.alpha3 + (v_vals - m_vals).pow(2)
        )
        
        loss_tm = torch.maximum(
            torch.zeros_like(m_vals),
            -args.alpha3 + (t_vals - m_vals).pow(2)
        )
        
        loss_vt = torch.maximum(
            torch.zeros_like(m_vals),
            -args.alpha3 + (v_vals - t_vals).pow(2)
        )
        
        plc_loss += (loss_vm.sum() + loss_tm.sum() + loss_vt.sum())
    
    plc_loss = plc_loss / (3 * batch_size * k)
    
    # 4. 模态对齐损失 (保留原代码逻辑)
    alignment_loss = modal_alignment_loss(visual_only1, text_only1, fused1)
    
    # 5. 跨模态对比损失 (保留原代码逻辑)
    v2t_loss = cross_modal_contrastive_loss(visual_only1, text_only2)
    
    # 6. 单模态到融合对比损失 (保留原代码逻辑)
    v2f_loss = contrastive_loss(visual_only1, fused2)
    t2f_loss = contrastive_loss(text_only1, fused2)
    
    # 7. 单模态间对比损失 (保留原代码逻辑)
    v2v_contra_loss = improved_contrastive_loss(
        visual_only1, visual_only2, 
        temperature=args.v2v_temperature,
        margin=0.2,  # 如果需要可以添加为参数
        hard_negative_ratio=0.5  # 如果需要可以添加为参数
    )
    t2t_contra_loss = contrastive_loss(text_only1, text_only2, temperature=0.03)
    
    # 加权组合得到总损失
    total_loss = (
        args.ppm_weight * ppm_loss + 
        args.pdc_weight * pdc_loss + 
        args.plc_weight * plc_loss +
        args.alignment_weight * alignment_loss +
        args.v2t_weight * v2t_loss +
        args.v2f_weight * v2f_loss +
        args.t2f_weight * t2f_loss +
        args.v2v_contra_weight * v2v_contra_loss +
        args.t2t_contra_weight * t2t_contra_loss
    ) / (args.ppm_weight + args.pdc_weight + args.plc_weight + 
         args.alignment_weight + args.v2t_weight + args.v2f_weight + 
         args.t2f_weight + args.v2v_contra_weight + args.t2t_contra_weight)
    
    # 构建指标字典
    metrics = {
        "ppm_loss": ppm_loss.item(),
        "pdc_loss": pdc_loss.item(),
        "plc_loss": plc_loss.item(),
        "alignment_loss": alignment_loss.item(),
        "v2t_loss": v2t_loss.item(),
        "v2f_loss": v2f_loss.item(),
        "t2f_loss": t2f_loss.item(),
        "v2v_contra_loss": v2v_contra_loss.item(),
        "t2t_contra_loss": t2t_contra_loss.item(),
        "total_loss": total_loss.item(),
        # 准确率指标
        "m2m_acc": sim_m2m.argmax(-1).eq(torch.arange(batch_size, device=sim_m2m.device)).float().mean().item(),
        "v2v_acc": sim_v2v.argmax(-1).eq(torch.arange(batch_size, device=sim_v2v.device)).float().mean().item(),
        "t2t_acc": sim_t2t.argmax(-1).eq(torch.arange(batch_size, device=sim_t2t.device)).float().mean().item(),
        "v2m_acc": sim_v2m.argmax(-1).eq(torch.arange(batch_size, device=sim_v2m.device)).float().mean().item(),
        "t2m_acc": sim_t2m.argmax(-1).eq(torch.arange(batch_size, device=sim_t2m.device)).float().mean().item()
    }
    
    return total_loss, metrics

# 通用对比损失函数
def contrastive_loss(feats1, feats2, temperature=0.07):
    """通用对比损失函数"""
    # 归一化特征
    feats1 = F.normalize(feats1, p=2, dim=1)
    feats2 = F.normalize(feats2, p=2, dim=1)
    
    # 计算相似度矩阵，使用传入的温度参数
    logits = torch.matmul(feats1, feats2.t()) / temperature
    
    # 对角线为正样本
    labels = torch.arange(len(feats1), device=feats1.device)
    
    # 计算损失
    loss = F.cross_entropy(logits, labels)
    
    return loss
    
####新加的
def freeze_vision_bn(args, model):
    # freeze bn running mean and variance
    if 'RN' in args.vision_model:
        RN_visual_modules = model.module.visual.modules() if isinstance(model, nn.parallel.DistributedDataParallel) else model.visual.modules()
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def train(model, data, epoch, optimizer, scaler, scheduler, args, global_trained_steps):
    loss_logger = None
    if is_master(args):
        log_dir = os.path.join(args.logspace, args.name, 'plots') if args.logspace else None
        loss_logger = LossLogger(log_dir)
    torch.cuda.empty_cache()
    gc.collect()
    # 基础设置
    model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data['train'].dataloader, data['train'].sampler
    # 使用交叉熵作为损失函数
    loss_fn = nn.CrossEntropyLoss().cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)
    
    # 初始化时间变量以修复end未定义的问题
    end = time.time()
    epoch_trained_steps = 0
    # 初始化large_batch_accs为None，确保它在首次使用时会被正确赋值
    accs_container = {
        'large_batch_accs': None
    }
    
    global_batch_accs = {
        "global_m2m_acc": 0.0,
        "global_v2v_acc": 0.0,
        "global_t2t_acc": 0.0,
        "global_v2m_acc": 0.0,
        "global_t2m_acc": 0.0,
        "global_v2t_acc": 0.0,
        "global_t2v_acc": 0.0
    }
    # 处理梯度累积
    if args.accum_freq > 1:
        # 用于收集特征和数据的列表
        accum_images1, accum_texts1 = [], []
        accum_images2, accum_texts2 = [], []
        accum_tags, accum_item_ids1, accum_item_ids2 = [], [], []
        
        # 用于收集特征的列表
        all_visual_only1, all_text_only1, all_fused1 = [], [], []
        all_visual_only2, all_text_only2, all_fused2 = [], [], []
        latest_logit_scale = None
    
    # 主训练循环
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)
        i_accum = i // args.accum_freq
        
        # 更新学习率调度
        step = num_steps_per_epoch * epoch + i_accum
        if step >= args.max_steps:
            logging.info(f"Stopping training due to step {step} has reached max_steps {args.max_steps}")
            return epoch_trained_steps
        scheduler(step)
    
        # 处理batch数据
        images1, texts1, eos_indices1, images2, texts2, eos_indices2, tags, item_ids1, item_ids2 = batch
        
        # 将所有数据移至GPU
        images1 = images1.cuda(args.local_device_rank, non_blocking=True)
        texts1 = texts1.cuda(args.local_device_rank, non_blocking=True)
        images2 = images2.cuda(args.local_device_rank, non_blocking=True)
        texts2 = texts2.cuda(args.local_device_rank, non_blocking=True)
        tags = tags.cuda(args.local_device_rank, non_blocking=True)
        item_ids1 = item_ids1.cuda(args.local_device_rank, non_blocking=True)
        item_ids2 = item_ids2.cuda(args.local_device_rank, non_blocking=True)
    
        data_time = time.time() - end
        batch_size = len(images1)
    
        # 不使用梯度累积的情况
        if args.accum_freq == 1:
            optimizer.zero_grad()
            # 原有代码保持不变
            # 反向传播使用全局损失
            if args.precision == "amp":
                optimizer.zero_grad()
                scaler.scale(global_total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                global_total_loss.backward()
                optimizer.step()
        else:
            # 使用梯度累积的情况 - 收集特征，不计算梯度
            # 使用梯度累积的情况 - 收集特征，保留梯度信息
            with autocast(enabled=(args.precision == "amp")):
                visual_only1, text_only1, fused1, visual_only2, text_only2, fused2, logit_scale = model(images1, texts1, images2, texts2)
                            
                # 保存logit_scale供大batch计算使用
                latest_logit_scale = logit_scale
                
                # 收集特征和数据
                all_visual_only1.append(visual_only1.clone())  # 使用clone而不是detach
                all_text_only1.append(text_only1.clone())
                all_fused1.append(fused1.clone())
                all_visual_only2.append(visual_only2.clone())
                all_text_only2.append(text_only2.clone())
                all_fused2.append(fused2.clone())
                
                # 收集原始数据以便可能需要的处理
                accum_images1.append(images1)
                accum_texts1.append(texts1)
                accum_images2.append(images2)
                accum_texts2.append(texts2)
                accum_tags.append(tags)
                accum_item_ids1.append(item_ids1)
                accum_item_ids2.append(item_ids2)
    
            # 如果还未累积够指定次数，继续下一个批次
            if ((i + 1) % args.accum_freq) > 0:
                continue
    
            # 累积够指定次数后，进行batch处理
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()
            # 连接特征形成GPU内累积batch
            large_batch_size = batch_size * args.accum_freq
            large_visual_only1 = torch.cat(all_visual_only1, dim=0)
            large_text_only1 = torch.cat(all_text_only1, dim=0)
            large_fused1 = torch.cat(all_fused1, dim=0)
            large_visual_only2 = torch.cat(all_visual_only2, dim=0)
            large_text_only2 = torch.cat(all_text_only2, dim=0)
            large_fused2 = torch.cat(all_fused2, dim=0)
            
            # 如果有多个GPU，收集所有GPU的特征以形成全局batch
            if args.world_size > 1:
                torch.cuda.empty_cache()
                gc.collect()
                # 使用带梯度的all_gather收集所有GPU的特征
                global_visual_only1 = all_gather_with_grad(large_visual_only1)
                global_text_only1 = all_gather_with_grad(large_text_only1)
                global_fused1 = all_gather_with_grad(large_fused1)
                global_visual_only2 = all_gather_with_grad(large_visual_only2)
                global_text_only2 = all_gather_with_grad(large_text_only2)
                global_fused2 = all_gather_with_grad(large_fused2)
                torch.cuda.empty_cache()
                gc.collect()
                # 更新全局批次大小
                global_batch_size = large_batch_size * args.world_size
                
                
                # 在全局batch上计算相似度矩阵和损失
                with autocast(enabled=(args.precision == "amp")):
                    # 计算全局batch上的相似度矩阵
                    global_sim_m2m = torch.matmul(global_fused1, global_fused2.t()) * latest_logit_scale
                    global_sim_v2v = torch.matmul(global_visual_only1, global_visual_only2.t()) * latest_logit_scale
                    global_sim_t2t = torch.matmul(global_text_only1, global_text_only2.t()) * latest_logit_scale
                    global_sim_v2m = torch.matmul(global_visual_only1, global_fused2.t()) * latest_logit_scale
                    global_sim_t2m = torch.matmul(global_text_only1, global_fused2.t()) * latest_logit_scale
                    global_sim_v2t = torch.matmul(global_visual_only1, global_text_only2.t()) * latest_logit_scale
                    
                    # 创建全局batch的标签矩阵和对角线索引
                    global_y_matrix = torch.eye(global_batch_size, device=global_sim_m2m.device)
                    global_diag_indices = torch.arange(global_batch_size, device=global_sim_m2m.device)
                    global_s_m2m_ii = global_sim_m2m[global_diag_indices, global_diag_indices].unsqueeze(1)
                    global_s_v2m_ii = global_sim_v2m[global_diag_indices, global_diag_indices].unsqueeze(1)
                    global_s_t2m_ii = global_sim_t2m[global_diag_indices, global_diag_indices].unsqueeze(1)
                    
                    # 1. PPM损失 - 在全局batch上计算
                    global_ppm_m2m_loss = torch.maximum(
                        torch.zeros_like(global_sim_m2m),
                        args.alpha1 * (1 - global_y_matrix) + global_sim_m2m - global_s_m2m_ii
                    ).sum()
                    
                    global_ppm_v2m_loss = torch.maximum(
                        torch.zeros_like(global_sim_v2m),
                        args.alpha1 * (1 - global_y_matrix) + global_sim_v2m - global_s_v2m_ii
                    ).sum()
                    
                    global_ppm_t2m_loss = torch.maximum(
                        torch.zeros_like(global_sim_t2m),
                        args.alpha1 * (1 - global_y_matrix) + global_sim_t2m - global_s_t2m_ii
                    ).sum()
                    
                    global_ppm_loss = (global_ppm_m2m_loss + global_ppm_v2m_loss + global_ppm_t2m_loss) / (3 * global_batch_size * global_batch_size)
                    
                    # 2. PDC损失 - 在全局batch上计算
                    global_pdc_v_loss = torch.maximum(
                        torch.zeros_like(global_sim_m2m),
                        args.alpha2 + global_sim_m2m - global_s_v2m_ii
                    ).sum()
                    
                    global_pdc_t_loss = torch.maximum(
                        torch.zeros_like(global_sim_m2m),
                        args.alpha2 + global_sim_m2m - global_s_t2m_ii
                    ).sum()
                    
                    global_pdc_loss = (global_pdc_v_loss + global_pdc_t_loss) / (2 * global_batch_size * global_batch_size)
                    
                    # 3. PLC损失 - 在全局batch上计算
                    k = min(10, global_batch_size)
                    _, top_indices = torch.topk(global_sim_m2m, k=k, dim=1)
                    
                    global_plc_loss = 0.0
                    for i_plc in range(global_batch_size):
                        indices = top_indices[i_plc]
                        m_vals = global_sim_m2m[i_plc, indices]
                        v_vals = global_sim_v2m[i_plc, indices]
                        t_vals = global_sim_t2m[i_plc, indices]
                        
                        loss_vm = torch.maximum(
                            torch.zeros_like(m_vals),
                            -args.alpha3 + (v_vals - m_vals).pow(2)
                        )
                        
                        loss_tm = torch.maximum(
                            torch.zeros_like(m_vals),
                            -args.alpha3 + (t_vals - m_vals).pow(2)
                        )
                        
                        loss_vt = torch.maximum(
                            torch.zeros_like(m_vals),
                            -args.alpha3 + (v_vals - t_vals).pow(2)
                        )
                        
                        global_plc_loss += (loss_vm.sum() + loss_tm.sum() + loss_vt.sum())
                    
                    global_plc_loss = global_plc_loss / (3 * global_batch_size * k)
                    
                    # 4. 模态对齐损失
                    global_alignment_loss = modal_alignment_loss(global_visual_only1, global_text_only1, global_fused1)
                    
                    # 5. 跨模态对比损失
                    global_v2t_loss = cross_modal_contrastive_loss(global_visual_only1, global_text_only2)
                    
                    # 6. 单模态到融合对比损失
                    global_v2f_loss = contrastive_loss(global_visual_only1, global_fused2)
                    global_t2f_loss = contrastive_loss(global_text_only1, global_fused2)
                    
                    # 7. 单模态间对比损失
                    global_v2v_contra_loss = improved_contrastive_loss(
                        global_visual_only1, global_visual_only2, 
                        temperature=args.v2v_temperature,
                        margin=0.2,
                        hard_negative_ratio=0.5
                    )
                    global_t2t_contra_loss = contrastive_loss(global_text_only1, global_text_only2, temperature=0.03)
                    
                    # 加权组合得到总损失
                    global_total_loss = (
                        args.ppm_weight * global_ppm_loss + 
                        args.pdc_weight * global_pdc_loss + 
                        args.plc_weight * global_plc_loss +
                        args.alignment_weight * global_alignment_loss +
                        args.v2t_weight * global_v2t_loss +
                        args.v2f_weight * global_v2f_loss +
                        args.t2f_weight * global_t2f_loss +
                        args.v2v_contra_weight * global_v2v_contra_loss +
                        args.t2t_contra_weight * global_t2t_contra_loss
                    ) / (args.ppm_weight + args.pdc_weight + args.plc_weight + 
                         args.alignment_weight + args.v2t_weight + args.v2f_weight + 
                         args.t2f_weight + args.v2v_contra_weight + args.t2t_contra_weight)
                    
                    # 构建指标字典用于日志记录
                    metrics = {
                        "ppm_loss": global_ppm_loss.item(),
                        "pdc_loss": global_pdc_loss.item(),
                        "plc_loss": global_plc_loss.item(),
                        "alignment_loss": global_alignment_loss.item(),
                        "v2t_loss": global_v2t_loss.item(),
                        "v2f_loss": global_v2f_loss.item(),
                        "t2f_loss": global_t2f_loss.item(),
                        "v2v_contra_loss": global_v2v_contra_loss.item(),
                        "t2t_contra_loss": global_t2t_contra_loss.item(),
                        "total_loss": global_total_loss.item(),
                        # 准确率指标
                        "m2m_acc": global_sim_m2m.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_m2m.device)).float().mean().item(),
                        "v2v_acc": global_sim_v2v.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_v2v.device)).float().mean().item(),
                        "t2t_acc": global_sim_t2t.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_t2t.device)).float().mean().item(),
                        "v2m_acc": global_sim_v2m.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_v2m.device)).float().mean().item(),
                        "t2m_acc": global_sim_t2m.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_t2m.device)).float().mean().item()
                    }
                
                # 更新参数 - 注意这里每个GPU使用相同的loss，不需要再平均
                if args.precision == "amp":
                    scaler.scale(global_total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    global_total_loss.backward()
                    optimizer.step()
                
                # 保存全局batch的准确率指标用于日志记录
                global_batch_accs = {
                    "global_m2m_acc": metrics["m2m_acc"],
                    "global_v2v_acc": metrics["v2v_acc"],
                    "global_t2t_acc": metrics["t2t_acc"],
                    "global_v2m_acc": metrics["v2m_acc"],
                    "global_t2m_acc": metrics["t2m_acc"],
                    "global_v2t_acc": global_sim_v2t.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_v2t.device)).float().mean().item(),
                    "global_t2v_acc": global_sim_t2t.argmax(-1).eq(torch.arange(global_batch_size, device=global_sim_t2t.device)).float().mean().item()
                }
                
                # 记录loss为全局batch的loss
                total_loss = global_total_loss
                
            else:
                # 如果只有单个GPU，使用大batch（而非全局batch）计算损失
                with autocast(enabled=(args.precision == "amp")):
                    # 计算大batch上的相似度矩阵
                    large_sim_m2m = torch.matmul(large_fused1, large_fused2.t()) * latest_logit_scale
                    large_sim_v2v = torch.matmul(large_visual_only1, large_visual_only2.t()) * latest_logit_scale
                    large_sim_t2t = torch.matmul(large_text_only1, large_text_only2.t()) * latest_logit_scale
                    large_sim_v2m = torch.matmul(large_visual_only1, large_fused2.t()) * latest_logit_scale
                    large_sim_t2m = torch.matmul(large_text_only1, large_fused2.t()) * latest_logit_scale
                    large_sim_v2t = torch.matmul(large_visual_only1, large_text_only2.t()) * latest_logit_scale
                    
                    # 创建大batch的标签矩阵和对角线索引
                    large_y_matrix = torch.eye(large_batch_size, device=large_sim_m2m.device)
                    large_diag_indices = torch.arange(large_batch_size, device=large_sim_m2m.device)
                    large_s_m2m_ii = large_sim_m2m[large_diag_indices, large_diag_indices].unsqueeze(1)
                    large_s_v2m_ii = large_sim_v2m[large_diag_indices, large_diag_indices].unsqueeze(1)
                    large_s_t2m_ii = large_sim_t2m[large_diag_indices, large_diag_indices].unsqueeze(1)
                    
                    # 1. PPM损失 - 在大batch上计算
                    large_ppm_m2m_loss = torch.maximum(
                        torch.zeros_like(large_sim_m2m),
                        args.alpha1 * (1 - large_y_matrix) + large_sim_m2m - large_s_m2m_ii
                    ).sum()
                    
                    large_ppm_v2m_loss = torch.maximum(
                        torch.zeros_like(large_sim_v2m),
                        args.alpha1 * (1 - large_y_matrix) + large_sim_v2m - large_s_v2m_ii
                    ).sum()
                    
                    large_ppm_t2m_loss = torch.maximum(
                        torch.zeros_like(large_sim_t2m),
                        args.alpha1 * (1 - large_y_matrix) + large_sim_t2m - large_s_t2m_ii
                    ).sum()
                    
                    large_ppm_loss = (large_ppm_m2m_loss + large_ppm_v2m_loss + large_ppm_t2m_loss) / (3 * large_batch_size * large_batch_size)
                    
                    # 2. PDC损失 - 在大batch上计算
                    large_pdc_v_loss = torch.maximum(
                        torch.zeros_like(large_sim_m2m),
                        args.alpha2 + large_sim_m2m - large_s_v2m_ii
                    ).sum()
                    
                    large_pdc_t_loss = torch.maximum(
                        torch.zeros_like(large_sim_m2m),
                        args.alpha2 + large_sim_m2m - large_s_t2m_ii
                    ).sum()
                    
                    large_pdc_loss = (large_pdc_v_loss + large_pdc_t_loss) / (2 * large_batch_size * large_batch_size)
                    
                    # 3. PLC损失 - 在大batch上计算
                    k = min(10, large_batch_size)
                    _, top_indices = torch.topk(large_sim_m2m, k=k, dim=1)
                    
                    large_plc_loss = 0.0
                    for i_plc in range(large_batch_size):
                        indices = top_indices[i_plc]
                        m_vals = large_sim_m2m[i_plc, indices]
                        v_vals = large_sim_v2m[i_plc, indices]
                        t_vals = large_sim_t2m[i_plc, indices]
                        
                        loss_vm = torch.maximum(
                            torch.zeros_like(m_vals),
                            -args.alpha3 + (v_vals - m_vals).pow(2)
                        )
                        
                        loss_tm = torch.maximum(
                            torch.zeros_like(m_vals),
                            -args.alpha3 + (t_vals - m_vals).pow(2)
                        )
                        
                        loss_vt = torch.maximum(
                            torch.zeros_like(m_vals),
                            -args.alpha3 + (v_vals - t_vals).pow(2)
                        )
                        
                        large_plc_loss += (loss_vm.sum() + loss_tm.sum() + loss_vt.sum())
                    
                    large_plc_loss = large_plc_loss / (3 * large_batch_size * k)
                    
                    # 4. 模态对齐损失
                    large_alignment_loss = modal_alignment_loss(large_visual_only1, large_text_only1, large_fused1)
                    
                    # 5. 跨模态对比损失
                    large_v2t_loss = cross_modal_contrastive_loss(large_visual_only1, large_text_only2)
                    
                    # 6. 单模态到融合对比损失
                    large_v2f_loss = contrastive_loss(large_visual_only1, large_fused2)
                    large_t2f_loss = contrastive_loss(large_text_only1, large_fused2)
                    
                    # 7. 单模态间对比损失
                    large_v2v_contra_loss = improved_contrastive_loss(
                        large_visual_only1, large_visual_only2, 
                        temperature=args.v2v_temperature,
                        margin=0.2,
                        hard_negative_ratio=0.5
                    )
                    large_t2t_contra_loss = contrastive_loss(large_text_only1, large_text_only2, temperature=0.03)
                    
                    # 加权组合得到总损失
                    large_total_loss = (
                        args.ppm_weight * large_ppm_loss + 
                        args.pdc_weight * large_pdc_loss + 
                        args.plc_weight * large_plc_loss +
                        args.alignment_weight * large_alignment_loss +
                        args.v2t_weight * large_v2t_loss +
                        args.v2f_weight * large_v2f_loss +
                        args.t2f_weight * large_t2f_loss +
                        args.v2v_contra_weight * large_v2v_contra_loss +
                        args.t2t_contra_weight * large_t2t_contra_loss
                    ) / (args.ppm_weight + args.pdc_weight + args.plc_weight + 
                         args.alignment_weight + args.v2t_weight + args.v2f_weight + 
                         args.t2f_weight + args.v2v_contra_weight + args.t2t_contra_weight)
                    
                    # 构建指标字典用于日志记录
                    metrics = {
                        "ppm_loss": large_ppm_loss.item(),
                        "pdc_loss": large_pdc_loss.item(),
                        "plc_loss": large_plc_loss.item(),
                        "alignment_loss": large_alignment_loss.item(),
                        "v2t_loss": large_v2t_loss.item(),
                        "v2f_loss": large_v2f_loss.item(),
                        "t2f_loss": large_t2f_loss.item(),
                        "v2v_contra_loss": large_v2v_contra_loss.item(),
                        "t2t_contra_loss": large_t2t_contra_loss.item(),
                        "total_loss": large_total_loss.item(),
                        # 准确率指标
                        "m2m_acc": large_sim_m2m.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_m2m.device)).float().mean().item(),
                        "v2v_acc": large_sim_v2v.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_v2v.device)).float().mean().item(),
                        "t2t_acc": large_sim_t2t.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_t2t.device)).float().mean().item(),
                        "v2m_acc": large_sim_v2m.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_v2m.device)).float().mean().item(),
                        "t2m_acc": large_sim_t2m.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_t2m.device)).float().mean().item()
                    }
                
                # 更新参数
                if args.precision == "amp":
                    scaler.scale(large_total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    large_total_loss.backward()
                    optimizer.step()
                
                # 记录large batch的准确率指标用于日志记录
                # 在计算大batch特征后，更新large_batch_accs
                accs_container['large_batch_accs'] = {
                    "large_m2m_acc": large_sim_m2m.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_m2m.device)).float().mean().item(),
                    "large_v2v_acc": large_sim_v2v.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_v2v.device)).float().mean().item(),
                    "large_t2t_acc": large_sim_t2t.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_t2t.device)).float().mean().item(),
                    "large_v2m_acc": large_sim_v2m.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_v2m.device)).float().mean().item(),
                    "large_t2m_acc": large_sim_t2m.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_t2m.device)).float().mean().item(),
                    "large_v2t_acc": large_sim_v2t.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_v2t.device)).float().mean().item(),
                    "large_t2v_acc": large_sim_t2t.argmax(-1).eq(torch.arange(large_batch_size, device=large_sim_t2t.device)).float().mean().item()
                }
                                                                
                # 记录loss为大batch的loss
                total_loss = large_total_loss
            
            # 清空累积的数据
            accum_images1, accum_texts1 = [], []
            accum_images2, accum_texts2 = [], []
            accum_tags = []
            accum_item_ids1 = []
            accum_item_ids2 = []
            all_visual_only1, all_text_only1, all_fused1 = [], [], []
            all_visual_only2, all_text_only2, all_fused2 = [], [], []
            torch.cuda.empty_cache()
            gc.collect()

        # 限制logit_scale的范围
        m = model.module
        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 3.0)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        # 记录训练信息
        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(images1) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            # 更新日志
            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Total Loss: {total_loss.item():.4f} | " +
                
                # 各组成部分损失
                f"PPM: {metrics['ppm_loss']:.4f} | " +
                f"PDC: {metrics['pdc_loss']:.4f} | " +
                f"PLC: {metrics['plc_loss']:.4f} | " +
                f"Align: {metrics['alignment_loss']:.4f} | " +
                f"V2T: {metrics['v2t_loss']:.4f} | " +
                f"V2F: {metrics['v2f_loss']:.4f} | " +
                f"T2F: {metrics['t2f_loss']:.4f} | " +
                f"V2V-Loss: {metrics['v2v_contra_loss']:.4f} | " +  
                f"T2T-Loss: {metrics['t2t_contra_loss']:.4f} | " +
                
                # 各类准确率
                f"M2M: {metrics['m2m_acc']*100:.2f}% | " +
                f"V2V: {metrics['v2v_acc']*100:.2f}% | " +
                f"T2T: {metrics['t2t_acc']*100:.2f}% | " +
                f"V2M: {metrics['v2m_acc']*100:.2f}% | " +
                f"T2M: {metrics['t2m_acc']*100:.2f}% | " +
                
                # 其他信息
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | " +
                f"Scale: {m.logit_scale.data:.3f} | " +
                f"Time: {batch_time:.3f}s"
            )
            if args.accum_freq > 1:
                if 'large_batch_accs' in accs_container and accs_container['large_batch_accs'] is not None:
                    logging.info(
                        f"Large Batch ({batch_size}) Accuracies: " +
                        f"M2M: {accs_container['large_batch_accs']['large_m2m_acc']*100:.2f}% | " +
                        f"V2V: {accs_container['large_batch_accs']['large_v2v_acc']*100:.2f}% | " +
                        f"T2T: {accs_container['large_batch_accs']['large_t2t_acc']*100:.2f}% | " +
                        f"V2M: {accs_container['large_batch_accs']['large_v2m_acc']*100:.2f}% | " +
                        f"T2M: {accs_container['large_batch_accs']['large_t2m_acc']*100:.2f}%"
                    )
                else:
                    logging.info(f"Large Batch accuracies not available (accs_container['large_batch_accs'] is None)")
            
            # 确保无论如何都打印global_batch信息（如果是多GPU）
            if args.world_size > 1:
                if 'global_batch_accs' in locals() and global_batch_accs is not None:
                    logging.info(
                        f"Global Batch ({batch_size*args.world_size}) Accuracies: " +
                        f"M2M: {global_batch_accs['global_m2m_acc']*100:.2f}% | " +
                        f"V2V: {global_batch_accs['global_v2v_acc']*100:.2f}% | " +
                        f"T2T: {global_batch_accs['global_t2t_acc']*100:.2f}% | " +
                        f"V2M: {global_batch_accs['global_v2m_acc']*100:.2f}% | " +
                        f"T2M: {global_batch_accs['global_t2m_acc']*100:.2f}%"
                    )
                else:
                    logging.info(f"Global Batch accuracies not available (global_batch_accs not defined)")
                    
            if is_master(args) and loss_logger is not None:
                try:
                    loss_logger.log_train(
                        total_loss=total_loss.item() if torch.is_tensor(total_loss) else total_loss,
                        ppm_loss=metrics['ppm_loss'],
                        pdc_loss=metrics['pdc_loss'],
                        plc_loss=metrics['plc_loss'],
                        acc=metrics['m2m_acc'],
                        step=step
                    )
                except Exception as e:
                    logging.warning(f"Failed to log training metrics: {str(e)}")

        # 验证
        if args.val_data is not None and args.valid_step_interval is not None and ((step + 1) % args.valid_step_interval) == 0:
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, step + 1)
            else:
                # fp16 is needed in flash attention
                with autocast():
                    evaluate(model, data, epoch, args, step + 1)
            # set model back to train mode
            torch.cuda.empty_cache()
            gc.collect()
            model.train()
            if args.freeze_vision:
                freeze_vision_bn(args, model)

        # 保存检查点
        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))

            # 保存最新检查点
            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1, step + 1, time.time() - t1))
            
    # 在训练循环结束时保存日志
    if is_master(args) and loss_logger is not None:
        try:
            loss_logger.plot_and_save()
        except Exception as e:
            logging.error(f"Failed to save training logs: {str(e)}")
    torch.cuda.empty_cache()
    gc.collect()      
    return epoch_trained_steps

# def evaluate(model, data, epoch, args, steps):
#     logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))
#     model.eval()
#     loss_logger = None
#     if is_master(args):
#         log_dir = os.path.join(args.logspace, args.name, 'plots') if args.logspace else None
#         loss_logger = LossLogger(log_dir)

#     dataloader = data['val'].dataloader
#     data_iter = iter(dataloader)
#     loss_fn = nn.CrossEntropyLoss().cuda(args.local_device_rank)
    
#     # 收集整个验证集的特征
#     all_visual_only1 = []
#     all_text_only1 = []
#     all_fused1 = []
#     all_visual_only2 = []
#     all_text_only2 = []
#     all_fused2 = []
#     latest_logit_scale = None
#     num_samples = 0
    
#     # 初始化累积指标 
#     cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
#     cumulative_ppm_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
#     cumulative_pdc_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
#     cumulative_plc_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
#     num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)

#     # 第一阶段：收集所有验证数据的特征
#     with torch.no_grad():
#         for i in range(dataloader.num_batches):
#             batch = next(data_iter)
#             images1, texts1, eos_indices1, images2, texts2, eos_indices2, tags, item_ids1, item_ids2 = batch
            
#             # 将数据移至GPU
#             images1 = images1.cuda(args.local_device_rank, non_blocking=True)
#             texts1 = texts1.cuda(args.local_device_rank, non_blocking=True)
#             images2 = images2.cuda(args.local_device_rank, non_blocking=True)
#             texts2 = texts2.cuda(args.local_device_rank, non_blocking=True)
#             tags = tags.cuda(args.local_device_rank, non_blocking=True)
            
#             # 获取模型输出
#             outputs = model(images1, texts1, images2, texts2)
#             visual_only1, text_only1, fused1, visual_only2, text_only2, fused2, logit_scale = outputs
            
#             # 保存logit_scale
#             latest_logit_scale = logit_scale
            
#             # 保存特征
#             all_visual_only1.append(visual_only1.cpu())
#             all_text_only1.append(text_only1.cpu())
#             all_fused1.append(fused1.cpu())
#             all_visual_only2.append(visual_only2.cpu())
#             all_text_only2.append(text_only2.cpu())
#             all_fused2.append(fused2.cpu())
            
#             batch_size = len(images1)
#             num_samples += batch_size
            
#             # 计算损失以保持与原来相同的累积指标
#             sim_m2m = torch.matmul(fused1, fused2.t()) * logit_scale
#             sim_v2m = torch.matmul(visual_only1, fused2.t()) * logit_scale
#             sim_t2m = torch.matmul(text_only1, fused2.t()) * logit_scale
            
#             ppm_loss = get_ppm_loss(visual_only1, text_only1, fused1, fused2, batch_size, args)
#             pdc_loss = get_pdc_loss(sim_m2m, sim_v2m, sim_t2m, batch_size, args)
#             plc_loss = get_plc_loss(sim_m2m, sim_v2m, sim_t2m, batch_size, args)
            
#             total_loss = (
#                 args.ppm_weight * ppm_loss + 
#                 args.pdc_weight * pdc_loss + 
#                 args.plc_weight * plc_loss
#             ) / (args.ppm_weight + args.pdc_weight + args.plc_weight)
            
#             # 累积计算
#             cumulative_loss += total_loss * batch_size
#             cumulative_ppm_loss += ppm_loss * batch_size
#             cumulative_pdc_loss += pdc_loss * batch_size
#             cumulative_plc_loss += plc_loss * batch_size
#             num_elements += batch_size

#             if (i + 1) % 100 == 0:
#                 logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))

#     # 将收集到的所有特征连接成单个张量
#     val_visual_only1 = torch.cat(all_visual_only1, dim=0).cuda(args.local_device_rank)
#     val_text_only1 = torch.cat(all_text_only1, dim=0).cuda(args.local_device_rank)
#     val_fused1 = torch.cat(all_fused1, dim=0).cuda(args.local_device_rank)
#     val_visual_only2 = torch.cat(all_visual_only2, dim=0).cuda(args.local_device_rank)
#     val_text_only2 = torch.cat(all_text_only2, dim=0).cuda(args.local_device_rank)
#     val_fused2 = torch.cat(all_fused2, dim=0).cuda(args.local_device_rank)
    
#     # 如果是多GPU环境，收集所有GPU上的特征
#     if args.world_size > 1:
#         # 获取每个GPU上的样本数量，以便正确构建全局索引
#         local_num_samples = torch.tensor(num_samples, device=val_fused1.device)
#         all_num_samples = [torch.zeros_like(local_num_samples) for _ in range(args.world_size)]
#         dist.all_gather(all_num_samples, local_num_samples)
        
#         # 创建收集所有GPU特征的列表
#         gathered_visual_only1 = [torch.zeros_like(val_visual_only1) for _ in range(args.world_size)]
#         gathered_text_only1 = [torch.zeros_like(val_text_only1) for _ in range(args.world_size)]
#         gathered_fused1 = [torch.zeros_like(val_fused1) for _ in range(args.world_size)]
#         gathered_visual_only2 = [torch.zeros_like(val_visual_only2) for _ in range(args.world_size)]
#         gathered_text_only2 = [torch.zeros_like(val_text_only2) for _ in range(args.world_size)]
#         gathered_fused2 = [torch.zeros_like(val_fused2) for _ in range(args.world_size)]
        
#         # 使用all_gather收集所有GPU的特征
#         dist.all_gather(gathered_visual_only1, val_visual_only1)
#         dist.all_gather(gathered_text_only1, val_text_only1)
#         dist.all_gather(gathered_fused1, val_fused1)
#         dist.all_gather(gathered_visual_only2, val_visual_only2)
#         dist.all_gather(gathered_text_only2, val_text_only2)
#         dist.all_gather(gathered_fused2, val_fused2)
        
#         # 连接所有GPU的特征
#         val_visual_only1 = torch.cat(gathered_visual_only1, dim=0)
#         val_text_only1 = torch.cat(gathered_text_only1, dim=0)
#         val_fused1 = torch.cat(gathered_fused1, dim=0)
#         val_visual_only2 = torch.cat(gathered_visual_only2, dim=0)
#         val_text_only2 = torch.cat(gathered_text_only2, dim=0)
#         val_fused2 = torch.cat(gathered_fused2, dim=0)
        
#         # 更新总样本数
#         num_samples = sum(s.item() for s in all_num_samples)
    
#     # 计算整个验证集上的相似度矩阵
#     val_sim_m2m = torch.matmul(val_fused1, val_fused2.t()) * latest_logit_scale
#     val_sim_v2v = torch.matmul(val_visual_only1, val_visual_only2.t()) * latest_logit_scale
#     val_sim_t2t = torch.matmul(val_text_only1, val_text_only2.t()) * latest_logit_scale
#     val_sim_v2m = torch.matmul(val_visual_only1, val_fused2.t()) * latest_logit_scale
#     val_sim_t2m = torch.matmul(val_text_only1, val_fused2.t()) * latest_logit_scale
#     val_sim_v2t = torch.matmul(val_visual_only1, val_text_only2.t()) * latest_logit_scale
#     val_sim_t2v = torch.matmul(val_text_only1, val_visual_only2.t()) * latest_logit_scale
    
#     # 创建正确答案索引
#     gt_indices = torch.arange(num_samples, device=val_sim_m2m.device)
    
#     # 计算Top-k准确率
#     def calculate_topk_accuracy(sim_matrix, k):
#         """计算Top-k准确率"""
#         topk_indices = torch.topk(sim_matrix, k=k, dim=-1)[1]
#         correct = (topk_indices == gt_indices.unsqueeze(-1)).any(dim=-1)
#         return correct.float().mean().item() * 100
    
#     # 计算各种模态组合的Top-k准确率
#     topk_values = [1, 5, 10, 20]
#     accuracy_metrics = {}
    
#     for k in topk_values:
#         accuracy_metrics[f"m2m_top{k}"] = calculate_topk_accuracy(val_sim_m2m, k)
#         accuracy_metrics[f"v2v_top{k}"] = calculate_topk_accuracy(val_sim_v2v, k)
#         accuracy_metrics[f"t2t_top{k}"] = calculate_topk_accuracy(val_sim_t2t, k)
#         accuracy_metrics[f"v2m_top{k}"] = calculate_topk_accuracy(val_sim_v2m, k)
#         accuracy_metrics[f"t2m_top{k}"] = calculate_topk_accuracy(val_sim_t2m, k)
#         accuracy_metrics[f"v2t_top{k}"] = calculate_topk_accuracy(val_sim_v2t, k)
#         accuracy_metrics[f"t2v_top{k}"] = calculate_topk_accuracy(val_sim_t2v, k)
    
#     # 同步多GPU结果 (原有累积计算的指标)
#     if args.world_size > 1:
#         dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
#         dist.all_reduce(cumulative_ppm_loss, op=dist.ReduceOp.SUM)
#         dist.all_reduce(cumulative_pdc_loss, op=dist.ReduceOp.SUM)
#         dist.all_reduce(cumulative_plc_loss, op=dist.ReduceOp.SUM)
#         dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)
    
#     # 计算平均值
#     avg_loss = cumulative_loss / num_elements
#     avg_ppm_loss = cumulative_ppm_loss / num_elements
#     avg_pdc_loss = cumulative_pdc_loss / num_elements
#     avg_plc_loss = cumulative_plc_loss / num_elements

#     # 记录验证结果
#     if is_master(args):
#         # 首先记录基本损失指标
#         logging.info(
#             f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
#             f"Valid Loss: {avg_loss.item():.6f} | "
#             f"PPM Loss: {avg_ppm_loss.item():.6f} | "
#             f"PDC Loss: {avg_pdc_loss.item():.6f} | "
#             f"PLC Loss: {avg_plc_loss.item():.6f} | "
#             f"logit_scale: {latest_logit_scale.item():.3f} | "
#             f"Total Validation Samples: {num_samples}"
#         )
        
#         # 然后记录所有Top-k准确率
#         for k in topk_values:
#             logging.info(
#                 f"Top-{k} Accuracy Metrics | "
#                 f"M2M: {accuracy_metrics[f'm2m_top{k}']:.2f}% | "
#                 f"V2V: {accuracy_metrics[f'v2v_top{k}']:.2f}% | "
#                 f"T2T: {accuracy_metrics[f't2t_top{k}']:.2f}% | "
#                 f"V2M: {accuracy_metrics[f'v2m_top{k}']:.2f}% | "
#                 f"T2M: {accuracy_metrics[f't2m_top{k}']:.2f}% | "
#                 f"V2T: {accuracy_metrics[f'v2t_top{k}']:.2f}% | "
#                 f"T2V: {accuracy_metrics[f't2v_top{k}']:.2f}%"
#             )
        
#         # 保存更细致的指标数据
#         if loss_logger is not None:
#             try:
#                 current_step = args.current_epoch * args.steps_per_epoch + steps
#                 loss_logger.log_val(
#                     total_loss=avg_loss.item(),
#                     ppm_loss=avg_ppm_loss.item(),
#                     pdc_loss=avg_pdc_loss.item(),
#                     plc_loss=avg_plc_loss.item(),
#                     top1_acc=accuracy_metrics['m2m_top1'],
#                     top5_acc=accuracy_metrics['m2m_top5'],
#                     top10_acc=accuracy_metrics['m2m_top10'],
#                     step=current_step
#                 )
#                 loss_logger.plot_and_save()
#             except Exception as e:
#                 logging.warning(f"Failed to save validation plots: {e}")
    
#     # 返回主要指标
#     return avg_loss.item(), accuracy_metrics['m2m_top1'] / 100.0  # 转换回0-1范围的准确率
def evaluate(model, data, epoch, args, steps):
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))
    model.eval()
    loss_logger = None
    if is_master(args):
        log_dir = os.path.join(args.logspace, args.name, 'plots') if args.logspace else None
        loss_logger = LossLogger(log_dir)

    dataloader = data['val'].dataloader
    data_iter = iter(dataloader)
    loss_fn = nn.CrossEntropyLoss().cuda(args.local_device_rank)
    
    # 创建保存嵌入向量的文件
    if is_master(args):
        emb_save_dir = os.path.join(args.logspace, args.name, 'embeddings')
        os.makedirs(emb_save_dir, exist_ok=True)
        emb_file1_path = os.path.join(emb_save_dir, f'item_id_1_embeddings_epoch{epoch+1}_step{steps}.jsonl')
        emb_file2_path = os.path.join(emb_save_dir, f'item_id_2_embeddings_epoch{epoch+1}_step{steps}.jsonl')
        emb_file1 = open(emb_file1_path, 'w', encoding='utf-8')
        emb_file2 = open(emb_file2_path, 'w', encoding='utf-8')
    
    # 初始化累积指标
    cumulative_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_ppm_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_pdc_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    cumulative_plc_loss = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    num_elements = torch.zeros([]).cuda(args.local_device_rank, non_blocking=True)
    num_samples = 0
    latest_logit_scale = None
    
    # 用于累积准确率指标的数据
    total_correct_m2m = {k: 0 for k in [1, 5, 10, 20]}
    total_correct_v2v = {k: 0 for k in [1, 5, 10, 20]}
    total_correct_t2t = {k: 0 for k in [1, 5, 10, 20]}
    total_correct_v2m = {k: 0 for k in [1, 5, 10, 20]}
    total_correct_t2m = {k: 0 for k in [1, 5, 10, 20]}
    total_correct_v2t = {k: 0 for k in [1, 5, 10, 20]}
    total_correct_t2v = {k: 0 for k in [1, 5, 10, 20]}

    # 逐批次处理验证数据
    with torch.no_grad():
        for i in range(dataloader.num_batches):
            batch = next(data_iter)
            images1, texts1, eos_indices1, images2, texts2, eos_indices2, tags, item_ids1, item_ids2 = batch
            
            # 将数据移至GPU
            images1 = images1.cuda(args.local_device_rank, non_blocking=True)
            texts1 = texts1.cuda(args.local_device_rank, non_blocking=True)
            images2 = images2.cuda(args.local_device_rank, non_blocking=True)
            texts2 = texts2.cuda(args.local_device_rank, non_blocking=True)
            tags = tags.cuda(args.local_device_rank, non_blocking=True)
            
            # 获取模型输出
            outputs = model(images1, texts1, images2, texts2)
            visual_only1, text_only1, fused1, visual_only2, text_only2, fused2, logit_scale = outputs
            latest_logit_scale = logit_scale
            
            batch_size = len(images1)
            num_samples += batch_size
            
            # 计算当前批次的相似度矩阵
            sim_m2m = torch.matmul(fused1, fused2.t()) * logit_scale
            sim_v2v = torch.matmul(visual_only1, visual_only2.t()) * logit_scale
            sim_t2t = torch.matmul(text_only1, text_only2.t()) * logit_scale
            sim_v2m = torch.matmul(visual_only1, fused2.t()) * logit_scale
            sim_t2m = torch.matmul(text_only1, fused2.t()) * logit_scale
            sim_v2t = torch.matmul(visual_only1, text_only2.t()) * logit_scale
            sim_t2v = torch.matmul(text_only1, visual_only2.t()) * logit_scale
            
            # 计算当前批次的Top-k准确率
            gt_indices = torch.arange(batch_size, device=sim_m2m.device)
            
            # 更新Top-k准确率累积计数
            for k in [1, 5, 10, 20]:
                total_correct_m2m[k] += (torch.topk(sim_m2m, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
                total_correct_v2v[k] += (torch.topk(sim_v2v, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
                total_correct_t2t[k] += (torch.topk(sim_t2t, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
                total_correct_v2m[k] += (torch.topk(sim_v2m, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
                total_correct_t2m[k] += (torch.topk(sim_t2m, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
                total_correct_v2t[k] += (torch.topk(sim_v2t, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
                total_correct_t2v[k] += (torch.topk(sim_t2v, k=k, dim=1)[1] == gt_indices.unsqueeze(-1)).any(dim=1).sum().item()
            
            # 如果是master节点，立即保存当前批次的嵌入向量
            if is_master(args):
                # 将特征移到CPU并转换为numpy数组
                visual_only1_cpu = visual_only1.cpu().numpy()
                text_only1_cpu = text_only1.cpu().numpy()
                fused1_cpu = fused1.cpu().numpy()
                visual_only2_cpu = visual_only2.cpu().numpy()
                text_only2_cpu = text_only2.cpu().numpy()
                fused2_cpu = fused2.cpu().numpy()
                item_ids1_cpu = item_ids1.cpu().numpy()
                item_ids2_cpu = item_ids2.cpu().numpy()
                
                # 逐条保存item_id_1的嵌入向量
                for j in range(batch_size):
                    data_dict = {
                        'item_id': int(item_ids1_cpu[j]),
                        'visual_embedding': visual_only1_cpu[j].tolist(),
                        'text_embedding': text_only1_cpu[j].tolist(),
                        'fusion_embedding': fused1_cpu[j].tolist()
                    }
                    emb_file1.write(json.dumps(data_dict) + '\n')
                
                # 逐条保存item_id_2的嵌入向量
                for j in range(batch_size):
                    data_dict = {
                        'item_id': int(item_ids2_cpu[j]),
                        'visual_embedding': visual_only2_cpu[j].tolist(),
                        'text_embedding': text_only2_cpu[j].tolist(),
                        'fusion_embedding': fused2_cpu[j].tolist()
                    }
                    emb_file2.write(json.dumps(data_dict) + '\n')
            
            # 计算当前批次的损失
            ppm_loss = get_ppm_loss(visual_only1, text_only1, fused1, fused2, batch_size, args)
            pdc_loss = get_pdc_loss(sim_m2m, sim_v2m, sim_t2m, batch_size, args)
            plc_loss = get_plc_loss(sim_m2m, sim_v2m, sim_t2m, batch_size, args)
            
            total_loss = (
                args.ppm_weight * ppm_loss + 
                args.pdc_weight * pdc_loss + 
                args.plc_weight * plc_loss
            ) / (args.ppm_weight + args.pdc_weight + args.plc_weight)
            
            # 累积计算
            cumulative_loss += total_loss * batch_size
            cumulative_ppm_loss += ppm_loss * batch_size
            cumulative_pdc_loss += pdc_loss * batch_size
            cumulative_plc_loss += plc_loss * batch_size
            num_elements += batch_size
            
            # 定期打印进度
            if (i + 1) % 100 == 0:
                logging.info("Evaluated {}/{} batches...".format(i + 1, dataloader.num_batches))
    
    # 关闭文件
    if is_master(args):
        emb_file1.close()
        emb_file2.close()
    
    # 如果使用分布式训练，需要同步各节点的累积指标
    if args.world_size > 1:
        dist.all_reduce(cumulative_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_ppm_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_pdc_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(cumulative_plc_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_elements, op=dist.ReduceOp.SUM)
        
        # 同步Top-k准确率计数
        for k in [1, 5, 10, 20]:
            count_tensor = torch.tensor([total_correct_m2m[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_m2m[k] = count_tensor.item()
            
            count_tensor = torch.tensor([total_correct_v2v[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_v2v[k] = count_tensor.item()
            
            count_tensor = torch.tensor([total_correct_t2t[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_t2t[k] = count_tensor.item()
            
            count_tensor = torch.tensor([total_correct_v2m[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_v2m[k] = count_tensor.item()
            
            count_tensor = torch.tensor([total_correct_t2m[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_t2m[k] = count_tensor.item()
            
            count_tensor = torch.tensor([total_correct_v2t[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_v2t[k] = count_tensor.item()
            
            count_tensor = torch.tensor([total_correct_t2v[k]], dtype=torch.float, device=cumulative_loss.device)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            total_correct_t2v[k] = count_tensor.item()
        
        # 同步样本总数
        samples_tensor = torch.tensor([num_samples], dtype=torch.float, device=cumulative_loss.device)
        dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
        num_samples = int(samples_tensor.item())
    
    # 计算平均损失
    avg_loss = cumulative_loss / num_elements
    avg_ppm_loss = cumulative_ppm_loss / num_elements
    avg_pdc_loss = cumulative_pdc_loss / num_elements
    avg_plc_loss = cumulative_plc_loss / num_elements
    
    # 计算Top-k准确率
    accuracy_metrics = {}
    for k in [1, 5, 10, 20]:
        accuracy_metrics[f"m2m_top{k}"] = (total_correct_m2m[k] / num_samples) * 100
        accuracy_metrics[f"v2v_top{k}"] = (total_correct_v2v[k] / num_samples) * 100
        accuracy_metrics[f"t2t_top{k}"] = (total_correct_t2t[k] / num_samples) * 100
        accuracy_metrics[f"v2m_top{k}"] = (total_correct_v2m[k] / num_samples) * 100
        accuracy_metrics[f"t2m_top{k}"] = (total_correct_t2m[k] / num_samples) * 100
        accuracy_metrics[f"v2t_top{k}"] = (total_correct_v2t[k] / num_samples) * 100
        accuracy_metrics[f"t2v_top{k}"] = (total_correct_t2v[k] / num_samples) * 100
    
    # 记录验证结果（与原代码相同）
    if is_master(args):
        logging.info(
            f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
            f"Valid Loss: {avg_loss.item():.6f} | "
            f"PPM Loss: {avg_ppm_loss.item():.6f} | "
            f"PDC Loss: {avg_pdc_loss.item():.6f} | "
            f"PLC Loss: {avg_plc_loss.item():.6f} | "
            f"logit_scale: {latest_logit_scale.item():.3f} | "
            f"Total Validation Samples: {num_samples}"
        )
        
        # 记录Top-k准确率
        for k in [1, 5, 10, 20]:
            logging.info(
                f"Top-{k} Accuracy Metrics | "
                f"M2M: {accuracy_metrics[f'm2m_top{k}']:.2f}% | "
                f"V2V: {accuracy_metrics[f'v2v_top{k}']:.2f}% | "
                f"T2T: {accuracy_metrics[f't2t_top{k}']:.2f}% | "
                f"V2M: {accuracy_metrics[f'v2m_top{k}']:.2f}% | "
                f"T2M: {accuracy_metrics[f't2m_top{k}']:.2f}% | "
                f"V2T: {accuracy_metrics[f'v2t_top{k}']:.2f}% | "
                f"T2V: {accuracy_metrics[f't2v_top{k}']:.2f}%"
            )
        
        # 保存验证日志
        if loss_logger is not None:
            try:
                current_step = args.current_epoch * args.steps_per_epoch + steps
                loss_logger.log_val(
                    total_loss=avg_loss.item(),
                    ppm_loss=avg_ppm_loss.item(),
                    pdc_loss=avg_pdc_loss.item(),
                    plc_loss=avg_plc_loss.item(),
                    top1_acc=accuracy_metrics['m2m_top1'],
                    top5_acc=accuracy_metrics['m2m_top5'],
                    top10_acc=accuracy_metrics['m2m_top10'],
                    step=current_step
                )
                loss_logger.plot_and_save()
            except Exception as e:
                logging.warning(f"Failed to save validation plots: {e}")
        
        logging.info(f"Successfully saved embeddings to {emb_file1_path} and {emb_file2_path}")
    
    # 返回主要指标
    return avg_loss.item(), accuracy_metrics['m2m_top1'] / 100.0
###新加的

def cosineSimilarityLoss(feature1, feature2):
    scale_factor_h = feature1.shape[0] / feature2.size(0)
    scale_factor_w = feature1.shape[1] / feature2.size(1)

    feature2_interpolated = F.interpolate(feature2.unsqueeze(0).unsqueeze(0),
                            size=(feature1.shape[0], feature1.shape[1]),
                            mode='bilinear',
                            align_corners=False)
    feature2_interpolated = feature2_interpolated.squeeze(0).squeeze(0)
    

    cosine_sim = F.cosine_similarity(feature1, feature2_interpolated, dim=1)
    similarity_loss = 1 - cosine_sim.mean()
    return similarity_loss
