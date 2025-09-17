from collections import OrderedDict
from typing import Tuple, Union
from itertools import repeat
import collections.abc

import math
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch import nn
from torch.utils.checkpoint import checkpoint

import importlib.util
if importlib.util.find_spec('flash_attn'):
    FlashMHA = importlib.import_module('flash_attn.flash_attention').FlashMHA

# from cn_clip.clip import _tokenizer
# from cn_clip.clip.configuration_bert import BertConfig
# from cn_clip.clip.modeling_bert import BertModel

from src.clip import _tokenizer
from src.clip.configuration_bert import BertConfig
from src.clip.modeling_bert import BertModel



class EnhancedCrossModalAttention(nn.Module):
    """改进的跨模态注意力模块，增加了门控机制和双向注意力"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 视觉->文本的注意力
        self.v2t_q_proj = nn.Linear(dim, dim)
        self.v2t_k_proj = nn.Linear(dim, dim)
        self.v2t_v_proj = nn.Linear(dim, dim)
        self.v2t_out_proj = nn.Linear(dim, dim)
        
        # 文本->视觉的注意力
        self.t2v_q_proj = nn.Linear(dim, dim)
        self.t2v_k_proj = nn.Linear(dim, dim)
        self.t2v_v_proj = nn.Linear(dim, dim)
        self.t2v_out_proj = nn.Linear(dim, dim)
        
        # 层标准化
        self.norm_v = nn.LayerNorm(dim)
        self.norm_t = nn.LayerNorm(dim)
        
        # 门控机制
        self.gate_v = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),  # +1 是为了接收存在性标志
            nn.Sigmoid()
        )
        self.gate_t = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),  # +1 是为了接收存在性标志
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def compute_attention(self, q_proj, k_proj, v_proj, out_proj, query, key, value):
        """计算注意力"""
        B, L_q, D = query.shape
        L_k = key.shape[1]
        H = self.num_heads
        
        q = q_proj(query).reshape(B, L_q, H, self.head_dim).transpose(1, 2)  # [B, H, L_q, d]
        k = k_proj(key).reshape(B, L_k, H, self.head_dim).transpose(1, 2)    # [B, H, L_k, d]
        v = v_proj(value).reshape(B, L_k, H, self.head_dim).transpose(1, 2)  # [B, H, L_k, d]
        
        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_k]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 计算输出
        out = (attn @ v).transpose(1, 2).reshape(B, L_q, D)  # [B, L_q, D]
        out = out_proj(out)
        out = self.dropout(out)
        
        return out
    
    def forward(self, visual_feats, text_feats):
        """前向传播，执行双向的跨模态注意力"""
        # 保存原始特征用于门控
        v_orig = visual_feats
        t_orig = text_feats
        
        v_norm = self.norm_v(visual_feats)
        t_norm = self.norm_t(text_feats)
        
        # 视觉关注文本
        v_attend_t = self.compute_attention(
            self.v2t_q_proj, self.v2t_k_proj, self.v2t_v_proj, self.v2t_out_proj,
            v_norm, t_norm, t_norm
        )
        
        # 文本关注视觉
        t_attend_v = self.compute_attention(
            self.t2v_q_proj, self.t2v_k_proj, self.t2v_v_proj, self.t2v_out_proj,
            t_norm, v_norm, v_norm
        )
        
        # 创建广播形状的存在性标志
        # 对整个文本序列求和判断是否存在（全局判断）
        t_exists_global = (torch.sum(torch.abs(t_norm)) > 1e-6).float()
        v_exists_global = (torch.sum(torch.abs(v_norm)) > 1e-6).float()
        
        # 将全局标志广播到与原特征相同的形状
        t_exists = torch.ones_like(v_orig[..., :1]) * t_exists_global
        v_exists = torch.ones_like(t_orig[..., :1]) * v_exists_global
        
        # 应用门控机制
        v_gate = self.gate_v(torch.cat([v_orig, v_attend_t, t_exists], dim=-1))
        t_gate = self.gate_t(torch.cat([t_orig, t_attend_v, v_exists], dim=-1))
        
        # 融合特征
        v_fused = v_gate * v_attend_t + (1 - v_gate) * v_orig
        t_fused = t_gate * t_attend_v + (1 - t_gate) * t_orig
        
        return v_fused, t_fused

class ImprovedMultiModalEncoder(nn.Module):
    """改进的多模态编码器，更强的特征交互和对齐，支持模态掩码"""
    def __init__(self, vision_width, text_width, embed_dim, num_cross_layers=3, num_fusion_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 投影层
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        
        # 增强的跨模态注意力层
        self.cross_layers = nn.ModuleList([
            EnhancedCrossModalAttention(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_cross_layers)
        ])
        
        # 单模态专注层（保留单模态信息）
        self.vision_self_attn = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(2)
        ])
        
        self.text_self_attn = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(2)
        ])
        
        # 池化层
        self.vision_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
        self.text_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
        # 使用Transformer块为融合网络
        self.fusion_layers = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, 
                num_heads=num_heads, 
                ffn_ratio=4, 
                dropout=dropout
            ) for _ in range(num_fusion_layers)
        ])
        
        # 融合前的映射
        self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # 额外的对齐层 - 确保单模态特征与融合特征对齐
        self.alignment_v2f = nn.Linear(embed_dim, embed_dim)
        self.alignment_t2f = nn.Linear(embed_dim, embed_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, image_features, text_features):
        """前向传播，保证所有输入都经过完整流程
        Args:
            image_features: [B, L_v, vision_width] 可能为全零（表示掩码）
            text_features: [B, L_t, text_width] 可能为全零（表示掩码）
                
        Returns:
            fusion_features: [B, embed_dim] - 融合特征
            vision_features: [B, embed_dim] - 视觉特征
            text_features: [B, embed_dim] - 文本特征
        """
        # 1. 投影到相同的维度空间
        img_feats = self.vision_proj(image_features)  
        txt_feats = self.text_proj(text_features)
        
        # 2. 增强的交叉模态交互 - 即使一个模态是零向量，也执行此步骤
        for layer in self.cross_layers:
            img_feats, txt_feats = layer(img_feats, txt_feats)
        
        # 3. 单模态特征增强 - 总是执行
        for layer in self.vision_self_attn:
            img_feats = layer(img_feats)
            
        for layer in self.text_self_attn:
            txt_feats = layer(txt_feats)
        
        # 4. 全局池化 - 获取序列级特征
        img_pooled = self.vision_pool(img_feats.mean(dim=1))  # [B, embed_dim]
        txt_pooled = self.text_pool(txt_feats.mean(dim=1))    # [B, embed_dim]
        
        # 5. 特征融合 - 不考虑是否为零向量
        combined = torch.cat([img_pooled, txt_pooled], dim=1)  # [B, embed_dim*2]
        fused = self.fusion_proj(combined).unsqueeze(1)  # [B, 1, embed_dim]
        
        # 通过Transformer层处理融合特征
        for layer in self.fusion_layers:
            fused = layer(fused)
        
        fusion_features = self.fusion_norm(fused.squeeze(1))  # [B, embed_dim]
        
        # 6. 单模态特征额外对齐 - 确保与融合特征在同一空间
        aligned_vision = self.alignment_v2f(img_pooled)
        aligned_text = self.alignment_t2f(txt_pooled)
        
        return fusion_features, aligned_vision, aligned_text

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Self-attention layers
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        B, L, D = x.shape
        H = self.num_heads
        
        q = self.q_proj(x).reshape(B, L, H, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, H, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        x = residual + self.gamma1 * out
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.gamma2 * self.ffn(x)
        
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            spacial_dim=input_resolution // 32,
            embed_dim=embed_dim, 
            num_heads=8,  # 固定使用8个头
            output_dim=output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def forward(self, x, return_all_tokens: bool = False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if return_all_tokens:
            # 不使用attnpool，而是直接将特征图展平为token序列
            b, c, h, w = x.shape
            spatial_tokens = x.reshape(b, c, h*w).permute(0, 2, 1)  # [B, h*w, c]
            
            # 添加一个额外的CLS token (使用全局平均池化)
            cls_token = torch.mean(spatial_tokens, dim=1, keepdim=True)  # [B, 1, c]
            tokens = torch.cat([cls_token, spatial_tokens], dim=1)  # [B, h*w+1, c]
            
            # 添加位置编码 (懒加载方式)
            if not hasattr(self, 'positional_embedding_for_tokens'):
                self.register_parameter(
                    'positional_embedding_for_tokens', 
                    nn.Parameter(torch.randn(h*w+1, c) / c**0.5)
                )
            
            tokens = tokens + self.positional_embedding_for_tokens.to(tokens.dtype)
            
            # 投影到目标维度
            if not hasattr(self, 'proj_for_tokens'):
                self.register_parameter(
                    'proj_for_tokens',
                    nn.Parameter(torch.randn(c, self.output_dim) / self.output_dim**0.5)
                )
            
            tokens = tokens @ self.proj_for_tokens.to(tokens.dtype)
            
            return tokens  # [B, h*w+1, output_dim]
        else:
            # 原始实现: 使用attnpool提取全局特征
            return self.attnpool(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) if not use_flash_attention else FlashMHA(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.use_flash_attention:
            # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x        
        return self.resblocks(x)

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_flash_attention: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device,
                                                               dtype=int)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False, mask_ratio: float = 0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if return_all_tokens:
            x = self.ln_post(x)  # 对所有token进行层归一化
            if self.proj is not None:
                # 对所有token进行投影
                x = x @ self.proj
            return x  # 返回所有token [B, 50, output_dim]
        else:
            # 原始实现：只返回 CLS token
            x = self.ln_post(x[:, 0, :])
            if self.proj is not None:
                x = x @ self.proj
            return x  # 只返回CLS token [B, output_dim]

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float, 
                 text_hidden_act: str, 
                 text_hidden_dropout_prob: float, 
                 text_hidden_size: int,
                 text_initializer_range: float, 
                 text_intermediate_size: int, 
                 text_max_position_embeddings: int, 
                 text_num_attention_heads: int, 
                 text_num_hidden_layers: int, 
                 text_type_vocab_size: int,
                 image_resolution: int = 224,
                 vision_layers: int = 12,
                 vision_width: int = 768,
                 vision_patch_size: int = 16,
                 vision_head_width: int = 64,
                 tokenizer = _tokenizer,
                 use_flash_attention: bool = False,
                 modality_mask_prob: float = 0.3):  # 添加模态掩码概率参数
        super().__init__()
        
        self.image_resolution = image_resolution
        self.modality_mask_prob = modality_mask_prob
            
        # 视觉编码器
        vision_heads = vision_width // 64
        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            use_flash_attention=use_flash_attention
        )

        # BERT文本编码器
        self.bert = BertModel(
            BertConfig(
                vocab_size_or_config_json_file=vocab_size,
                hidden_size=text_hidden_size,
                num_hidden_layers=text_num_hidden_layers,
                num_attention_heads=text_num_attention_heads,
                intermediate_size=text_intermediate_size,
                hidden_act=text_hidden_act,
                hidden_dropout_prob=text_hidden_dropout_prob,
                attention_probs_dropout_prob=text_attention_probs_dropout_prob,
                max_position_embeddings=text_max_position_embeddings,
                type_vocab_size=text_type_vocab_size,
                initializer_range=text_initializer_range,
                layer_norm_eps=1e-12,
                use_flash_attention=use_flash_attention
            )
        )
        
        # 文本投影层
        self.text_projection = nn.Parameter(torch.empty(text_hidden_size, embed_dim))
        
        # 多模态融合编码器
        self.multimodal_encoder = ImprovedMultiModalEncoder(
            vision_width=embed_dim,
            text_width=embed_dim,
            embed_dim=embed_dim,
            num_cross_layers=3,
            num_fusion_layers=3,
            num_heads=8,
            dropout=text_hidden_dropout_prob
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        self.tokenizer = tokenizer

        self.initialize_parameters()

    def normalize_features(self, features):
        """将特征向量L2归一化"""
        if isinstance(features, tuple):
            features = features[0]
        return F.normalize(features, p=2, dim=-1)
    
    def encode_image(self, image, return_all_tokens: bool = False):
        """编码并归一化图像特征"""
        features = self.visual(image.type(self.dtype), return_all_tokens=return_all_tokens)
        
        if return_all_tokens:
            features_shape = features.shape
            features = features.reshape(-1, features.size(-1))
            features = self.normalize_features(features)
            features = features.reshape(features_shape)
            return features
        else:
            return self.normalize_features(features)
    
    def encode_text_all_tokens(self, text):
        """编码并归一化文本特征，保留所有token"""
        pad_index = self.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.dtype)
        x = self.bert(text, attention_mask=attn_mask)[0].type(self.dtype)
        return self.normalize_features(x @ self.text_projection)
    
    def forward(self, images1, texts1, images2=None, texts2=None):
        """
        前向传播函数，支持单组或双组输入，同时为每种输入生成三种特征表示
        
        Args:
            images1: 第一组图像输入，形状为[B, 3, H, W]
            texts1: 第一组文本输入，形状为[B, L]
            images2: 第二组图像输入，形状为[B, 3, H, W]，默认为None
            texts2: 第二组文本输入，形状为[B, L]，默认为None
        
        Returns:
            多种特征表示的元组
        """
        device = self.logit_scale.device
        batch_size1 = max(images1.size(0) if images1 is not None else 0, 
                          texts1.size(0) if texts1 is not None else 0)
        
        # 1. 生成第一组多模态特征
        visual_out1 = self.encode_image(images1, return_all_tokens=True)
        text_out1 = self.encode_text_all_tokens(texts1)
        if text_out1.size(1) < 64:
            pad_len = 64 - text_out1.size(1)
            text_out1 = F.pad(text_out1, (0, 0, 0, pad_len))
        elif text_out1.size(1) > 64:
            text_out1 = text_out1[:, :64, :]
        
        fusion_features1, aligned_vision1, aligned_text1 = self.multimodal_encoder(visual_out1, text_out1)
        
        # 2. 生成第一组纯图像特征 (文本置0)
        zero_text_out1 = torch.zeros_like(text_out1, device=device)
        visual_only_features1, _, _ = self.multimodal_encoder(visual_out1, zero_text_out1)
        
        # 3. 生成第一组纯文本特征 (图像置0)
        zero_visual_out1 = torch.zeros_like(visual_out1, device=device)
        text_only_features1, _, _ = self.multimodal_encoder(zero_visual_out1, text_out1)
        
        # 标准归一化
        fusion_features1 = F.normalize(fusion_features1, p=2, dim=1)
        visual_only_features1 = F.normalize(visual_only_features1, p=2, dim=1)
        text_only_features1 = F.normalize(text_only_features1, p=2, dim=1)
        
        # 如果没有提供第二组输入，只返回第一组结果
        if images2 is None or texts2 is None:
            return visual_only_features1, text_only_features1, fusion_features1, self.logit_scale.exp()
        
        # 处理第二组输入
        batch_size2 = max(images2.size(0) if images2 is not None else 0, 
                          texts2.size(0) if texts2 is not None else 0)
        
        # 1. 生成第二组多模态特征
        visual_out2 = self.encode_image(images2, return_all_tokens=True)
        text_out2 = self.encode_text_all_tokens(texts2)
        if text_out2.size(1) < 64:
            pad_len = 64 - text_out2.size(1)
            text_out2 = F.pad(text_out2, (0, 0, 0, pad_len))
        elif text_out2.size(1) > 64:
            text_out2 = text_out2[:, :64, :]
        
        fusion_features2, aligned_vision2, aligned_text2 = self.multimodal_encoder(visual_out2, text_out2)
        
        # 2. 生成第二组纯图像特征 (文本置0)
        zero_text_out2 = torch.zeros_like(text_out2, device=device)
        visual_only_features2, _, _ = self.multimodal_encoder(visual_out2, zero_text_out2)
        
        # 3. 生成第二组纯文本特征 (图像置0)
        zero_visual_out2 = torch.zeros_like(visual_out2, device=device)
        text_only_features2, _, _ = self.multimodal_encoder(zero_visual_out2, text_out2)
        
        # 标准归一化
        fusion_features2 = F.normalize(fusion_features2, p=2, dim=1)
        visual_only_features2 = F.normalize(visual_only_features2, p=2, dim=1)
        text_only_features2 = F.normalize(text_only_features2, p=2, dim=1)
        
        # 返回两组结果以及logit_scale
        return visual_only_features1, text_only_features1, fusion_features1, \
               visual_only_features2, text_only_features2, fusion_features2, \
               self.logit_scale.exp()
        

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            return next(self.parameters()).dtype

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.bert.config.hidden_size ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.bert.set_grad_checkpointing(enable)

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        if isinstance(l, BertModel):
            l.to(torch.half)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def restore_model(model, clip_state_dict: dict, bert_state_dict: dict, use_flash_attention: bool):
    merged_state_dict = {}

    # use clip_state_dict to initialize the image encoder & logit scale
    if clip_state_dict is not None:
        for k, v in clip_state_dict.items():
            if k.startswith("visual") or k == "logit_scale":
                merged_state_dict[k] = v

    # use bert_state_dict to initialize the text encoder
    if bert_state_dict is not None:
        for k, v in bert_state_dict.items():
            if k.startswith("bert") and "bert.pooler" not in k:
                merged_state_dict[k] = v

    # adapt flash attention
    if use_flash_attention:
        merged_state_dict = convert_state_dict(merged_state_dict)

    convert_weights(model)
    resize_pos_embed(merged_state_dict, model)
    model.load_state_dict(merged_state_dict, strict=False)
    return model.eval()


def convert_state_dict(state_dict):
    """Adapt to Flash Attention"""
    if not state_dict:
        return state_dict

    prefix = 'module.' if list(state_dict.keys())[0].startswith('module') else ''

    if f'{prefix}visual.transformer.resblocks.0.attn.in_proj_weight' in state_dict:
        for k in list(state_dict.keys()):
            if 'attn.in_proj_weight' in k:
                state_dict[k.replace('attn.in_proj_weight', 'attn.Wqkv.weight')] = state_dict.pop(k)
            elif 'attn.in_proj_bias' in k:
                state_dict[k.replace('attn.in_proj_bias', 'attn.Wqkv.bias')] = state_dict.pop(k)
    elif f'{prefix}visual.transformer.resblocks.0.attn.Wqkv.weight' in state_dict:
        for k in list(state_dict.keys()):
            if 'attn.Wqkv.weight' in k:
                state_dict[k.replace('attn.Wqkv.weight', 'attn.in_proj_weight')] = state_dict.pop(k)
            elif 'attn.Wqkv.bias' in k:
                state_dict[k.replace('attn.Wqkv.bias', 'attn.in_proj_bias')] = state_dict.pop(k)

    if f'{prefix}bert.encoder.layer.0.attention.self.query.weight' in state_dict:
        i = 0
        while f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight' in state_dict:
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'] = torch.cat(
                (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'))
            )
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'] = torch.cat(
                (state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'),
                 state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'))
            )
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight')
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.bias'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias')
            i += 1
    elif f'{prefix}bert.encoder.layer.0.attention.self.Wqkv.weight' in state_dict:
        i = 0
        while f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight' in state_dict:
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.weight'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.weight'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.weight'] = \
                torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.weight'), chunks=3)
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.query.bias'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.key.bias'], \
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.self.value.bias'] = \
                torch.chunk(state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.Wqkv.bias'), chunks=3)
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.weight'] = \
                state_dict.pop(f'{prefix}bert.encoder.layer.{i}.attention.self.out_proj.weight')
            state_dict[f'{prefix}bert.encoder.layer.{i}.attention.output.dense.bias'] = \
                state_dict.pop(f'module.bert.encoder.layer.{i}.attention.self.out_proj.bias')
            i += 1

    return state_dict


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1, prefix=""):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get(prefix + 'visual.positional_embedding', None)
    model = model.module if hasattr(model, 'module') else model
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + 'visual.positional_embedding'] = new_pos_embed


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)