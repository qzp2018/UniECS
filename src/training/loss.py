import torch
import torch.nn.functional as F

def negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale):
    # Normalize embeddings
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()
    img_negtext_similarity = logit_scale * img_embs @ neg_text_embs.t()

    preds_i2t = torch.cat((img_text_similarity, img_negtext_similarity), dim=-1).argmax(
        dim=-1
    )
    preds_t2i = img_text_similarity.t().argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    loss = (
        F.cross_entropy(
            torch.cat([img_text_similarity, img_negtext_similarity], dim=-1), labels
        )
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    return loss, accuracy


def tripletclip_loss(img_embs, text_embs, neg_img_embs, neg_text_embs, logit_scale):
    loss_1, accuracy1 = negclip_loss(img_embs, text_embs, neg_text_embs, logit_scale)
    loss_2, accuracy2 = negclip_loss(neg_img_embs, neg_text_embs, text_embs, logit_scale)

    loss = loss_1 + loss_2
    accuracy = (accuracy1 + accuracy2) / 2
    return loss, accuracy

def clip_loss(img_embs, text_embs, logit_scale):

    # Normalize embeddings
    batch_size = img_embs.shape[0]
    labels = torch.arange(batch_size, device=img_embs.device).long()

    img_text_similarity = logit_scale * img_embs @ text_embs.t()
    text_img_similarity = logit_scale * text_embs @ img_embs.t()

    preds_i2t = img_text_similarity.argmax(dim=-1)
    preds_t2i = text_img_similarity.argmax(dim=-1)
    acc_i2t = (preds_i2t == labels).float().mean().item()
    acc_t2i = (preds_t2i == labels).float().mean().item()
    accuracy = (acc_i2t + acc_t2i) / 2

    loss = (
        F.cross_entropy(img_text_similarity, labels)
        + F.cross_entropy(text_img_similarity, labels)
    ).div(2)
    return loss, accuracy


def get_loss_simple_w_hinge(model, images, texts, loss_img, loss_txt, args, accum_image_features=None, accum_text_features=None, accum_idx=-1, teacher_model=None, teacher_accum_image_features=None):
    image_features, text_features, logit_scale = model(images, texts, args.mask_ratio)


    logit_scale = logit_scale.mean()

    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)

    total_loss = (
        loss_img(logits_per_image, ground_truth)
        + loss_txt(logits_per_text, ground_truth)
    ) / 2

    # 计算文本模态的对比损失
    # 我们假设正样本和负样本之间的距离通过一定的策略进行区分
    text_contrastive_loss = 0
    if args.text_contrastive_loss_weight > 0:
        # 对比损失的计算，假设我们将文本特征进行对比
        # 正样本与负样本的区分需要有额外的标签信息（可以是分类标签等）
        positive_text_features = text_features[ground_truth]  # 正样本：ground_truth对应的特征
        negative_text_features = text_features  # 负样本：其余的特征

        text_contrastive_loss = contrastive_loss(text_features, positive_text_features, negative_text_features, margin=1.0)


    # 添加文本模态对比损失
    total_loss += text_contrastive_loss * args.text_contrastive_loss_weight

    acc = None
    return total_loss, acc

############################
#https://github.com/vinid/neg_clip
# 不兼容分布式ing
# 1修改数据加载方式（负样本）
# 2修改前向传递
# 3 修改loss



###################
###借鉴阿里ben神论文里的loss
import torch
import torch.nn.functional as F

# Get the tensor shape list
def get_shape_list(tensor):
    shape_list = list(tensor.shape)
    return shape_list

# Product to product matching losses (PPM)
def hinge_batch_loss(trigger_emb, offer_emb, margin, batch_size):
    # Normalize embeddings
    trigger_emb_norm = torch.max(torch.norm(trigger_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=trigger_emb.device))
    trigger_emb_repr = trigger_emb / trigger_emb_norm
    offer_emb_norm = torch.max(torch.norm(offer_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=offer_emb.device))
    offer_emb_repr = offer_emb / offer_emb_norm

    # Cosine similarity matrix
    dis = torch.matmul(trigger_emb_repr, offer_emb_repr.T)
    positive_distance = torch.diag(dis).view(batch_size, 1)
    
    # Hinge loss calculation
    eye_mask = torch.eye(batch_size, device=dis.device)
    hinge_loss = torch.mean(torch.max(torch.zeros_like(dis), -positive_distance + dis + margin * (1 - eye_mask)), dim=1)
    
    return hinge_loss

# Product self-distinctiveness losses (PDC)
def self_distinctiveness_loss(trigger_emb, trigger_trans_emb, offer_emb, margin):
    # Normalize embeddings
    trigger_emb_norm = torch.max(torch.norm(trigger_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=trigger_emb.device))
    trigger_emb_rep = trigger_emb / trigger_emb_norm
    trigger_trans_emb_norm = torch.max(torch.norm(trigger_trans_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=trigger_trans_emb.device))
    trigger_trans_emb_rep = trigger_trans_emb / trigger_trans_emb_norm
    offer_emb_norm = torch.max(torch.norm(offer_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=offer_emb.device))
    offer_emb_rep = offer_emb / offer_emb_norm
    batch_size = get_shape_list(trigger_emb)[0]

    # Cosine similarity matrix
    dis = torch.matmul(trigger_trans_emb_rep, trigger_emb_rep.T)
    positive_distance = torch.diag(dis).view(batch_size, 1)
    
    # Self distinctiveness loss calculation
    eye_mask = torch.eye(batch_size, device=dis.device)
    self_distinct_loss = torch.mean(torch.max(torch.zeros_like(dis), -positive_distance + dis + margin * (1 - eye_mask)), dim=1)
    
    return self_distinct_loss

# Locality consistency loss
def locality_consistency_loss(trigger_emb, trigger_trans_emb, offer_emb, margin):
    # Normalize embeddings
    trigger_emb_norm = torch.max(torch.norm(trigger_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=trigger_emb.device))
    trigger_emb_repr = trigger_emb / trigger_emb_norm
    trigger_trans_emb_norm = torch.max(torch.norm(trigger_trans_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=trigger_trans_emb.device))
    trigger_trans_emb_repr = trigger_trans_emb / trigger_trans_emb_norm
    offer_emb_norm = torch.max(torch.norm(offer_emb, dim=1, keepdim=True), torch.tensor(1e-12, device=offer_emb.device))
    offer_emb_repr = offer_emb / offer_emb_norm

    # Cosine similarity matrix
    dis = torch.matmul(trigger_emb_repr, offer_emb_repr.T)
    trans_dis = torch.matmul(trigger_trans_emb_repr, offer_emb_repr.T)

    # Only get top10 similar products for compute economy
    batch_size = get_shape_list(trigger_emb)[0]
    values, indices = torch.topk(dis, k=10, dim=1, largest=True, sorted=False)
    
    idx_flattened = (torch.arange(batch_size, device=dis.device).view(-1, 1) * batch_size + indices).view(-1)
    gathered_values = torch.gather((dis - trans_dis)**2, 0, idx_flattened)
    gathered_values = gathered_values.view(batch_size, 10)
    
    # Locality consistency loss calculation
    locality_consistency_loss = torch.mean(torch.max(torch.zeros_like(gathered_values), gathered_values - margin), dim=1)
    
    return locality_consistency_loss
