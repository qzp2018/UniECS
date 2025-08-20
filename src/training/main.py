from math import ceil
import os
import logging
from pathlib import Path
import json
import time
from time import gmtime, strftime
import importlib.util
import numpy
import torch.serialization
import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from src.clip import load
from src.clip.model import convert_weights, convert_state_dict, resize_pos_embed, CLIP#
from src.training.train import train, evaluate
from src.training.data import get_data
from src.training.params import parse_args
from src.training.logger import setup_primary_logging, setup_worker_logging
from src.training.scheduler import cosine_lr

import matplotlib.pyplot as plt
import numpy as np
# 在 main.py 文件开头添加:
global loss_logger
loss_logger = None
_loss_logger = None
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
    
    def log_train(self, total_loss, ppm_loss, pdc_loss, plc_loss, acc, step):
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
# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()


def is_master(args):
    return args.rank == 0


# used to compare the pytorch version
def torch_version_str_compare_lessequal(version1, version2):
    v1 = [int(entry) for entry in version1.split("+")[0].split(".")]
    v2 = [int(entry) for entry in version2.split("+")[0].split(".")]
    assert len(v1) == 3, "Cannot parse the version of your installed pytorch! ({})".format(version1)
    assert len(v2) == 3, "Illegal version specification ({}). Should be in 1.X.Y format.".format(version2)
    return sorted([v1, v2])[0] == v1


def main():
    try:
        global _loss_logger
        log_queue = None
        args = parse_args()
        args.current_epoch = 0
        args.steps_per_epoch = 0

        # Set distributed group
        args.local_device_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_device_rank)
        args.device = torch.device("cuda", args.local_device_rank)

        dist.init_process_group(backend="nccl")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        # Set output path
        time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        args.log_path = os.path.join(args.logspace, args.name, "out_{}.log".format(time_suffix))
        args.checkpoint_path = os.path.join(args.logspace, args.name, "checkpoints") 
        args.plot_path = os.path.join(args.logspace, args.name, "plots")
        
        # 创建所有必要的目录 
        if is_master(args):
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
            os.makedirs(args.checkpoint_path, exist_ok=True)
            os.makedirs(args.plot_path, exist_ok=True)  
            _loss_logger = LossLogger(args.plot_path)
            # 新增：打印training_history.npy的存储路径
            logging.info(f"LossLogger初始化完成，训练历史数据将保存至: {_loss_logger.log_dir}")
            _loss_logger.load_history()

        # Set logger
        args.log_level = logging.DEBUG if args.debug else logging.INFO
        log_queue = setup_primary_logging(args.log_path, args.log_level, args.rank)
        setup_worker_logging(args.rank, log_queue, args.log_level)

        # Build the CLIP model
        vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
        text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
        
        with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
            model_info = json.load(fv)
            if isinstance(model_info['vision_layers'], str):
                model_info['vision_layers'] = eval(model_info['vision_layers'])         
            for k, v in json.load(ft).items():
                model_info[k] = v

        model = CLIP(**model_info)

        # Load pre-trained weights
        if args.clip_weight_path is not None:
            assert os.path.exists(args.clip_weight_path), "Pretrained CLIP weight not exists!"
        if args.bert_weight_path is not None:
            assert os.path.exists(args.bert_weight_path), "Pretrained BERT weight not exists!"
        load(model, clip_path=args.clip_weight_path, bert_path=args.bert_weight_path)

        # Convert model precision
        if args.precision == "amp" or args.precision == "fp32":
            convert_models_to_fp32(model)

        model.cuda(args.local_device_rank)
        if args.precision == "fp16":
            convert_weights(model)

        if args.grad_checkpointing:
            model.set_grad_checkpointing()
            logging.info("Grad-checkpointing activated.")

        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if args.freeze_vision:
            for k, v in model.visual.named_parameters():
                v.requires_grad = False
            if args.vision_model in ['RN50']:
                for m in model.visual.modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()
            logging.info("The visual encoder is freezed during training.")

        # DDP setup
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_device_rank],
            find_unused_parameters=True
        )

        if args.precision == "fp16":
            convert_weights(model)

        # Initialize dataset and dataloader
        data = get_data(args, epoch_id=0, max_txt_length=args.context_length)
        args.steps_per_epoch = data["train"].dataloader.num_batches

        # Initialize optimizer and scheduler
        exclude = lambda n : "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n : not exclude(n)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n) and p.requires_grad]

        if args.train_data is None:
            optimizer = None
            scheduler = None
        else:
            # 将参数分为视觉编码器参数和其他参数
            vision_gain_or_bias_params = [p for n, p in named_parameters if "visual" in n and exclude(n) and p.requires_grad]
            vision_rest_params = [p for n, p in named_parameters if "visual" in n and include(n) and p.requires_grad]
            other_gain_or_bias_params = [p for n, p in named_parameters if "visual" not in n and exclude(n) and p.requires_grad]
            other_rest_params = [p for n, p in named_parameters if "visual" not in n and include(n) and p.requires_grad]
            
            optimizer = optim.AdamW(
                [
                    {"params": vision_gain_or_bias_params, "weight_decay": 0., "lr": args.vision_lr},
                    {"params": vision_rest_params, "weight_decay": args.wd, "lr": args.vision_lr},
                    {"params": other_gain_or_bias_params, "weight_decay": 0., "lr": args.lr},
                    {"params": other_rest_params, "weight_decay": args.wd, "lr": args.lr},
                ],
                lr=args.lr,  # 这个是默认学习率
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
            
            num_batches = data["train"].dataloader.num_batches
            if args.max_steps is not None:
                args.max_epochs = ceil(args.max_steps / num_batches)
            else:
                assert args.max_epochs is not None and args.max_epochs > 0
                args.max_steps = num_batches * args.max_epochs
                
            total_steps = args.max_steps
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

        scaler = GradScaler() if args.precision == "amp" else None

        # Log parameters
        if is_master(args):
            logging.info("Params:")
            params_file = os.path.join(args.logspace, args.name, "params_{}.txt".format(time_suffix))
            with open(params_file, "w", encoding="utf-8") as f:
                for name in sorted(vars(args)):
                    val = getattr(args, name)
                    f.write(f"{name}: {val}\n")

        # Main training loop
        start_epoch = 0
        steps = 0

        # Resume from checkpoint if specified
        if args.resume is None:
            latest_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            if os.path.isfile(latest_path):
                args.resume = latest_path

        if args.resume is not None:
            if os.path.isfile(args.resume):
                
                try:
                    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
                except AttributeError:
                    # 对于新版PyTorch，这个方法不存在，可以跳过
                    pass
                checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
                sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
                
                # 获取检查点中保存的门控网络输入维度（如果有的话）
                gate_input_dim = checkpoint.get("model_config", {}).get("gate_input_dim", 512)
                current_gate_input_dim = 513  # 当前模型的门控网络输入维度
                
                # 处理门控网络参数不匹配问题
                for i in range(3):  # 3个cross_layers层
                    # 处理视觉门控
                    v_key = f"module.multimodal_encoder.cross_layers.{i}.gate_v.0.weight"
                    if v_key in sd:
                        old_weight = sd[v_key]
                        old_dim = old_weight.shape[1]
                        
                        if old_dim < current_gate_input_dim:
                            # 如果检查点中的权重更小，扩展它
                            new_weight = torch.zeros((256, current_gate_input_dim), device=old_weight.device, dtype=old_weight.dtype)
                            new_weight[:, :old_dim] = old_weight
                            sd[v_key] = new_weight
                        elif old_dim > current_gate_input_dim:
                            # 如果检查点中的权重更大，截取它
                            sd[v_key] = old_weight[:, :current_gate_input_dim]
                    
                    # 处理文本门控  
                    t_key = f"module.multimodal_encoder.cross_layers.{i}.gate_t.0.weight"
                    if t_key in sd:
                        old_weight = sd[t_key]
                        old_dim = old_weight.shape[1]
                        
                        if old_dim < current_gate_input_dim:
                            # 如果检查点中的权重更小，扩展它
                            new_weight = torch.zeros((256, current_gate_input_dim), device=old_weight.device, dtype=old_weight.dtype)
                            new_weight[:, :old_dim] = old_weight
                            sd[t_key] = new_weight
                        elif old_dim > current_gate_input_dim:
                            # 如果检查点中的权重更大，截取它
                            sd[t_key] = old_weight[:, :current_gate_input_dim]
                
                # 加载处理过的状态字典
                model.load_state_dict(sd, strict=False)
                
                # 其余代码保持不变
                if not args.reset_data_offset:
                    start_epoch = checkpoint["epoch"]
                    steps = checkpoint["step"]
                    data = get_data(args, epoch_id=start_epoch, max_txt_length=args.context_length)
                
                if not args.reset_optimizer and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    
                logging.info(f"Resumed from checkpoint: {args.resume}")
            else:
                logging.info("No checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True
        cudnn.deterministic = False

        # Determine if this worker should save logs/checkpoints
        args.should_save = (args.logspace is not None and args.logspace != '' and 
                          args.logspace.lower() != 'none') and is_master(args)

        # Training epochs
        for epoch in range(start_epoch, args.max_epochs):
            args.current_epoch = epoch
            try:
                if is_master(args):
                    logging.info(f'Start epoch {epoch + 1}')
                    
                num_steps_this_epoch = train(model, data, epoch, optimizer, 
                           scaler, scheduler, args, steps)
                steps += num_steps_this_epoch
    
                # Validation
                if (args.val_data is not None and args.valid_epoch_interval is not None and 
                    ((epoch + 1) % args.valid_epoch_interval) == 0):
                    evaluate(model, data, epoch, args, steps)
    
                # Prepare next epoch's data
                if epoch + 1 < args.max_epochs:
                    data = get_data(args, epoch_id=epoch+1, max_txt_length=args.context_length)
    
                # Save checkpoints
                if args.should_save and num_steps_this_epoch > 0:
                    # Save epoch checkpoint if needed
                    if ((epoch + 1) == args.max_epochs or 
                        (args.save_epoch_frequency > 0 and 
                         ((epoch + 1) % args.save_epoch_frequency) == 0)):
                        # 在main.py中找到保存checkpoint的代码段
                        save_path = os.path.join(args.checkpoint_path, f"epoch{epoch + 1}.pt")
                        # 创建一个包含所有配置的字典
                        model_config = {
                            # 基本模型架构参数
                            "embed_dim": args.embed_dim if hasattr(args, 'embed_dim') else 256,
                            "image_resolution": model_info["image_resolution"],
                            "vision_layers": model_info["vision_layers"],
                            "vision_width": model_info["vision_width"],
                            "vision_patch_size": 16 if args.vision_model == "ViT-B-16" else 32,
                            "context_length": args.context_length,
                            
                            # 训练参数（可能对推理有用）
                            "vision_model": args.vision_model,
                            "text_model": args.text_model,
                            
                            # 损失函数权重
                            "alignment_weight": args.alignment_weight,
                            "v2t_weight": args.v2t_weight,
                            "v2f_weight": args.v2f_weight,
                            "t2f_weight": args.t2f_weight,
                            "v2v_contra_weight": args.v2v_contra_weight,
                            "v2v_temperature": args.v2v_temperature,
                            
                            # 门控网络配置 - 添加这一行来记录门控网络的输入维度
                            "gate_input_dim": 513  # 明确记录门控网络的输入维度
                        }
                        
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "step": steps,
                                "name": args.name,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "model_config": model_config,  # 保存模型配置
                                "args": args,  # 也可以直接保存整个args对象
                            },
                            save_path,
                        )
    
                    # Always save latest checkpoint
                    # 在main.py中找到保存checkpoint的代码段
                    save_path = os.path.join(args.checkpoint_path, f"epoch{epoch + 1}.pt")
                    # 创建一个包含所有配置的字典
                    model_config = {
                        # 基本模型架构参数
                        "embed_dim": args.embed_dim if hasattr(args, 'embed_dim') else 256,
                        "image_resolution": model_info["image_resolution"],
                        "vision_layers": model_info["vision_layers"],
                        "vision_width": model_info["vision_width"],
                        "vision_patch_size": 16 if args.vision_model == "ViT-B-16" else 32,
                        "context_length": args.context_length,
                        
                        # 训练参数（可能对推理有用）
                        "vision_model": args.vision_model,
                        "text_model": args.text_model,
                        
                        # 损失函数权重
                        "alignment_weight": args.alignment_weight,
                        "v2t_weight": args.v2t_weight,
                        "v2f_weight": args.v2f_weight,
                        "t2f_weight": args.t2f_weight,
                        "v2v_contra_weight": args.v2v_contra_weight,
                        "v2v_temperature": args.v2v_temperature,
                        
                        # 门控网络配置 - 添加这一行来记录门控网络的输入维度
                        "gate_input_dim": 513  # 明确记录门控网络的输入维度
                    }
                    
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "step": steps,
                            "name": args.name,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "model_config": model_config,  # 保存模型配置
                            "args": args,  # 也可以直接保存整个args对象
                        },
                        save_path,
                    )
            except Exception as e:
                logging.error(f"Error in epoch {epoch + 1}: {str(e)}", exc_info=True)
                if dist.is_initialized():
                    dist.destroy_process_group()
                raise e

        # 在训练结束时保存日志
        if is_master(args) and _loss_logger is not None:
            _loss_logger.plot_and_save()
            
        # 清理分布式进程组
        if dist.is_initialized():
            dist.destroy_process_group()

    except Exception as e:
        logging.error("Error occurred during execution:", exc_info=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e
    finally:
        # 替换原有的清理代码为：
        if dist.is_initialized():
            dist.destroy_process_group()
        
        # 新增以下日志清理代码
        if log_queue is not None:
            log_queue.put_nowait(None)
            time.sleep(0.1)
            
        for handler in logging.getLogger().handlers[:]:
            try:
                handler.close()
                logging.getLogger().removeHandler(handler)
            except:
                pass

if __name__ == "__main__":
    main()