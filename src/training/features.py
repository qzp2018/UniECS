# 特征融合
# 备战特征缺失状态
# 图文-图文
# 图-图文
# 文-图
# 多阶段训练，课程学习

import torch
import torch.nn as nn



class MLPFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, image_features, text_features):
        fused_features = torch.cat([image_features, text_features], dim=-1)
        x = self.fc1(fused_features)
        x = self.relu(x)
        return self.fc2(x)




def feature_merge(image_features,text_features,merge_func):
    #merge_func 可以选几种不同的融合函数

    #判断纬度大小
    batch_size = image_features.shape[0]
    embed_dim = image_features.shape[-1]

    # 缺失模态处理
    if image_features is None:
        fused_features = text_features
    elif text_features is None:
        fused_features = image_features
    else:
        fused_features = merge_func(image_features, text_features)

    # Example usage
    fuse_model = MLPFusion(input_dim=image_features.shape[-1] + text_features.shape[-1], hidden_dim=256, output_dim=128)
    fused_features = fuse_model(image_features, text_features)

    return fused_features





# Concatenate features
def fuse_features_concat(image_features, text_features):
    return torch.cat([image_features, text_features], dim=-1)


# Add features
def fuse_features_add(image_features, text_features):
    return image_features + text_features


# Multiply features
def fuse_features_multiply(image_features, text_features):
    return image_features * text_features



def fuse_features_bilinear(image_features, text_features):
    return torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)).flatten(start_dim=1)




'''
如何生成图片和文本的特征向量。
如何融合多模态特征成统一的多模态向量。      满足对缺失模态的鲁棒性。简单加权融合\学习融合方法（MLP 或 Transformer）
如何处理缺失模态（如只有图片或只有文本）。    对于缺失模态，用零向量（或均值向量）填充，或者只用非缺失模态特征
如何设计和训练检索模型。

根据多模态（图片 + 文本）特征融合向量，训练一个相似商品检索系统。
多模态向量通过对比学习或分类学习，确保相似商品的特征向量距离更近。


对于同一个商品 ID 的多条记录，视为正样本。
随机采样不同商品 ID 作为负样本。

'''



