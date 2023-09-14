import torch
import torch.nn as nn
import random
import torchvision
import numpy as np


def get_content_extension_loss(feats_s, feats_sw, feats_w, gts, queue):
    B, C, H, W = feats_s.shape  # feat feature size (B X C X H X W)

    # uniform sampling with a size of 64*64 for source and wild-stylized source feature maps
    H_W_resize = 64
    HW = H_W_resize * H_W_resize

    upsample_n = nn.Upsample(size=[H_W_resize, H_W_resize], mode='nearest')
    feats_s_flat = upsample_n(feats_s)
    feats_sw_flat = upsample_n(feats_sw)

    feats_s_flat = feats_s_flat.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    feats_sw_flat = feats_sw_flat.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    gts_flat = upsample_n(gts.unsqueeze(1).float()).squeeze(1).long().view(B, HW)

    # uniform sampling with a size of small scale for wild feature map
    H_W_resize_w = 36
    HW_w = H_W_resize_w * H_W_resize_w

    upsample_n_w = nn.Upsample(size=[H_W_resize_w, H_W_resize_w], mode='nearest')
    feats_w_flat = upsample_n_w(feats_w)
    feats_w_flat = torch.einsum("bchw->bhwc", feats_w_flat).contiguous().view(B * H_W_resize_w * H_W_resize_w,
                                                                              C)  # B X C X H X W > (B X H X W) X C

    # normalize feature of each pixel
    feats_s_flat = nn.functional.normalize(feats_s_flat, p=2, dim=1)
    feats_sw_flat = nn.functional.normalize(feats_sw_flat, p=2, dim=1)
    feats_w_flat = nn.functional.normalize(feats_w_flat, p=2, dim=1).detach()  # (B X H X W) X C

    # log(dot(feats_s_flat, feats_sw_flat))
    T = 0.07
    logits_sce = torch.bmm(feats_s_flat.transpose(1, 2), feats_sw_flat)  # dot product: B X (H X W) X (H X W)
    logits_sce = (torch.clamp(logits_sce, min=-1, max=1)) / T

    # compute ignore mask: same-class (excluding self) + unknown-labeled pixels
    # compute positive mask (same-class)
    logits_mask_sce_ignore = torch.eq(gts_flat.unsqueeze(2),
                                      gts_flat.unsqueeze(1))  # pos:1, neg:0. B X (H X W) X (H X W)
    # include unknown-labeled pixel
    logits_mask_sce_ignore = include_unknown(logits_mask_sce_ignore, gts_flat)

    # exclude self-pixel
    logits_mask_sce_ignore *= ~torch.eye(HW, HW).type(torch.cuda.BoolTensor).unsqueeze(0).expand(
        [B, -1, -1])  # self:1, other:0. B X (H X W) X (H X W)

    # compute positive mask for cross entropy loss: B X (H X W)
    logits_mask_sce_pos = torch.linspace(start=0, end=HW - 1, steps=HW).unsqueeze(0).expand([B, -1]).type(
        torch.cuda.LongTensor)

    # compute unknown-labeled mask for cross entropy loss: B X (H X W)
    logits_mask_sce_unk = torch.zeros_like(logits_mask_sce_pos, dtype=torch.bool)
    logits_mask_sce_unk[gts_flat > 254] = True

    # compute loss_sce
    eps = 1e-5
    logits_sce[logits_mask_sce_ignore] = -1 / T
    CELoss = nn.CrossEntropyLoss(reduction='none')
    loss_sce = CELoss(logits_sce.transpose(1, 2), logits_mask_sce_pos)
    loss_sce = ((loss_sce * (~logits_mask_sce_unk)).sum(1) / ((~logits_mask_sce_unk).sum(1) + eps)).mean()

    # get wild content closest to wild-stylized source content
    idx_sim_bs = 512
    index_nearest_neighbours = (torch.randn(0)).type(torch.cuda.LongTensor)
    for idx_sim in range(int(np.ceil(HW / idx_sim_bs))):
        idx_sim_start = idx_sim * idx_sim_bs
        idx_sim_end = min((idx_sim + 1) * idx_sim_bs, HW)
        similarity_matrix = torch.einsum("bcn,cq->bnq",
                                         feats_sw_flat[:, :, idx_sim_start:idx_sim_end].type(torch.cuda.HalfTensor),
                                         queue['wild'].type(torch.cuda.HalfTensor))  # B X (H X W) X Q
        index_nearest_neighbours = torch.cat((index_nearest_neighbours, torch.argmax(similarity_matrix, dim=2)),
                                             dim=1)  # B X (H X W)
    # similarity_matrix = torch.einsum("bcn,cq->bnq", feats_sw_flat, queue['wild']) # B X (H X W) X Q
    # index_nearest_neighbours = torch.argmax(similarity_matrix, dim=2) # B X (H X W)
    del similarity_matrix
    nearest_neighbours = torch.index_select(queue['wild'], dim=1, index=index_nearest_neighbours.view(-1)).view(C, B,
                                                                                                                HW)  # C X B X (H X W)

    # compute exp(dot(feats_s_flat, nearest_neighbours))
    logits_wce_pos = torch.einsum("bcn,cbn->bn", feats_s_flat,
                                  nearest_neighbours)  # dot product: B X C X (H X W) & C X B X (H X W) => B X (H X W)
    logits_wce_pos = (torch.clamp(logits_wce_pos, min=-1, max=1)) / T
    exp_logits_wce_pos = torch.exp(logits_wce_pos)

    # compute negative mask of logits_sce
    logits_mask_sce_neg = ~torch.eq(gts_flat.unsqueeze(2), gts_flat.unsqueeze(1))  # pos:0, neg:1. B X (H X W) X (H X W)

    # exclude unknown-labeled pixels from negative samples
    logits_mask_sce_neg = exclude_unknown(logits_mask_sce_neg, gts_flat)

    # sum exp(neg samples)
    exp_logits_sce_neg = (torch.exp(logits_sce) * logits_mask_sce_neg).sum(2)  # B X (H X W)

    # Compute log_prob
    log_prob_wce = logits_wce_pos - torch.log(exp_logits_wce_pos + exp_logits_sce_neg)  # B X (H X W)

    # Compute loss_wce
    loss_wce = -((log_prob_wce * (~logits_mask_sce_unk)).sum(1) / ((~logits_mask_sce_unk).sum(1) + eps)).mean()

    # enqueue wild contents
    sup_enqueue = feats_w_flat  # Q X C # (B X H X W) X C
    _dequeue_and_enqueue(queue, sup_enqueue)

    # compute content extension learning loss
    loss_cel = loss_sce +  loss_wce

    return loss_cel

def small_scale_get_content_extension_loss(feats_s, feats_sw, feats_w, gts, queue):
    B, C, H, W = feats_s.shape  # feat feature size (B X C X H X W)

    # uniform sampling with a size of large_scale for source and wild-stylized source feature maps
    H_W_resize = 128
    HW = H_W_resize * H_W_resize

    upsample_n = nn.Upsample(size=[H_W_resize, H_W_resize], mode='nearest')
    feats_s_flat = upsample_n(feats_s)
    feats_sw_flat = upsample_n(feats_sw)

    feats_s_flat = feats_s_flat.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    feats_sw_flat = feats_sw_flat.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    gts_flat = upsample_n(gts.unsqueeze(1).float()).squeeze(1).long().view(B, HW)

    # uniform sampling with a size of large_scale for wild feature map
    H_W_resize_w = 72
    HW_w = H_W_resize_w * H_W_resize_w

    upsample_n_w = nn.Upsample(size=[H_W_resize_w, H_W_resize_w], mode='nearest')
    feats_w_flat = upsample_n_w(feats_w)
    feats_w_flat = torch.einsum("bchw->bhwc", feats_w_flat).contiguous().view(B * H_W_resize_w * H_W_resize_w,
                                                                              C)  # B X C X H X W > (B X H X W) X C

    # normalize feature of each pixel
    feats_s_flat = nn.functional.normalize(feats_s_flat, p=2, dim=1)
    feats_sw_flat = nn.functional.normalize(feats_sw_flat, p=2, dim=1)
    feats_w_flat = nn.functional.normalize(feats_w_flat, p=2, dim=1).detach()  # (B X H X W) X C

    # log(dot(feats_s_flat, feats_sw_flat))
    T = 0.07
    logits_sce = torch.bmm(feats_s_flat.transpose(1, 2), feats_sw_flat)  # dot product: B X (H X W) X (H X W)
    logits_sce = (torch.clamp(logits_sce, min=-1, max=1)) / T

    # compute ignore mask: same-class (excluding self) + unknown-labeled pixels
    # compute positive mask (same-class)
    logits_mask_sce_ignore = torch.eq(gts_flat.unsqueeze(2),
                                      gts_flat.unsqueeze(1))  # pos:1, neg:0. B X (H X W) X (H X W)
    # include unknown-labeled pixel
    logits_mask_sce_ignore = include_unknown(logits_mask_sce_ignore, gts_flat)

    # exclude self-pixel
    logits_mask_sce_ignore *= ~torch.eye(HW, HW).type(torch.cuda.BoolTensor).unsqueeze(0).expand(
        [B, -1, -1])  # self:1, other:0. B X (H X W) X (H X W)

    # compute positive mask for cross entropy loss: B X (H X W)
    logits_mask_sce_pos = torch.linspace(start=0, end=HW - 1, steps=HW).unsqueeze(0).expand([B, -1]).type(
        torch.cuda.LongTensor)

    # compute unknown-labeled mask for cross entropy loss: B X (H X W)
    logits_mask_sce_unk = torch.zeros_like(logits_mask_sce_pos, dtype=torch.bool)
    logits_mask_sce_unk[gts_flat > 254] = True

    # compute loss_sce
    eps = 1e-5
    logits_sce[logits_mask_sce_ignore] = -1 / T
    CELoss = nn.CrossEntropyLoss(reduction='none')
    loss_sce = CELoss(logits_sce.transpose(1, 2), logits_mask_sce_pos)
    loss_sce = ((loss_sce * (~logits_mask_sce_unk)).sum(1) / ((~logits_mask_sce_unk).sum(1) + eps)).mean()

    # get wild content closest to wild-stylized source content
    idx_sim_bs = 512
    index_nearest_neighbours = (torch.randn(0)).type(torch.cuda.LongTensor)
    for idx_sim in range(int(np.ceil(HW / idx_sim_bs))):
        idx_sim_start = idx_sim * idx_sim_bs
        idx_sim_end = min((idx_sim + 1) * idx_sim_bs, HW)
        similarity_matrix = torch.einsum("bcn,cq->bnq",
                                         feats_sw_flat[:, :, idx_sim_start:idx_sim_end].type(torch.cuda.HalfTensor),
                                         queue['wild'].type(torch.cuda.HalfTensor))  # B X (H X W) X Q
        index_nearest_neighbours = torch.cat((index_nearest_neighbours, torch.argmax(similarity_matrix, dim=2)),
                                             dim=1)  # B X (H X W)
    # similarity_matrix = torch.einsum("bcn,cq->bnq", feats_sw_flat, queue['wild']) # B X (H X W) X Q
    # index_nearest_neighbours = torch.argmax(similarity_matrix, dim=2) # B X (H X W)
    del similarity_matrix
    nearest_neighbours = torch.index_select(queue['wild'], dim=1, index=index_nearest_neighbours.view(-1)).view(C, B,
                                                                                                                HW)  # C X B X (H X W)

    # compute exp(dot(feats_s_flat, nearest_neighbours))
    logits_wce_pos = torch.einsum("bcn,cbn->bn", feats_s_flat,
                                  nearest_neighbours)  # dot product: B X C X (H X W) & C X B X (H X W) => B X (H X W)
    logits_wce_pos = (torch.clamp(logits_wce_pos, min=-1, max=1)) / T
    exp_logits_wce_pos = torch.exp(logits_wce_pos)

    # compute negative mask of logits_sce
    logits_mask_sce_neg = ~torch.eq(gts_flat.unsqueeze(2), gts_flat.unsqueeze(1))  # pos:0, neg:1. B X (H X W) X (H X W)

    # exclude unknown-labeled pixels from negative samples
    logits_mask_sce_neg = exclude_unknown(logits_mask_sce_neg, gts_flat)

    # sum exp(neg samples)
    exp_logits_sce_neg = (torch.exp(logits_sce) * logits_mask_sce_neg).sum(2)  # B X (H X W)

    # Compute log_prob
    log_prob_wce = logits_wce_pos - torch.log(exp_logits_wce_pos + exp_logits_sce_neg)  # B X (H X W)

    # Compute loss_wce
    loss_wce = -((log_prob_wce * (~logits_mask_sce_unk)).sum(1) / ((~logits_mask_sce_unk).sum(1) + eps)).mean()

    # enqueue wild contents
    sup_enqueue = feats_w_flat  # Q X C # (B X H X W) X C
    _dequeue_and_enqueue(queue, sup_enqueue)

    # compute content extension learning loss
    loss_cel = loss_sce +  loss_wce

    return loss_cel


def exclude_unknown(mask, gts):
    '''
    mask: [B, HW, HW]
    gts: [B, HW]
    '''
    mask = mask.transpose(1, 2).contiguous()
    mask[gts > 254, :] = False
    mask = mask.transpose(1, 2).contiguous()

    return mask


def include_unknown(mask, gts):
    '''
    mask: [B, HW, HW]
    gts: [B, HW]
    '''
    mask = mask.transpose(1, 2).contiguous()
    mask[gts > 254, :] = True
    mask = mask.transpose(1, 2).contiguous()

    return mask


@torch.no_grad()
def _dequeue_and_enqueue(queue, keys):
    # gather keys before updating queue
    # keys = concat_all_gather(keys)  # (B X H X W) X C

    batch_size = keys.shape[0]

    ptr = int(queue['wild_ptr'])

    # replace the keys at ptr (dequeue and enqueue)
    if (ptr + batch_size) <= queue['size']:
        # wild queue
        queue['wild'][:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % queue['size']  # move pointer
    else:
        # wild queue
        last_input_num = queue['size'] - ptr
        queue['wild'][:, ptr:] = (keys.T)[:, :last_input_num]
        ptr = (ptr + batch_size) % queue['size']  # move pointer
        queue['wild'][:, :ptr] = (keys.T)[:, last_input_num:]
    queue['wild_ptr'][0] = ptr


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = varsize_tensor_all_gather(tensor)

    output = tensors_gather
    return output


def varsize_tensor_all_gather(tensor: torch.Tensor):
    tensor = tensor.contiguous()

    cuda_device = f'cuda:{torch.distributed.get_rank()}'
    size_tens = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=cuda_device)

    size_tens2 = [torch.ones_like(size_tens)
                  for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(size_tens2, size_tens)
    size_tens2 = torch.cat(size_tens2, dim=0).cpu()
    max_size = size_tens2.max()

    padded = torch.empty(max_size, *tensor.shape[1:],
                         dtype=tensor.dtype,
                         device=cuda_device)
    padded[:tensor.shape[0]] = tensor

    ag = [torch.ones_like(padded)
          for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(ag, padded)
    ag = torch.cat(ag, dim=0)

    slices = []
    for i, sz in enumerate(size_tens2):
        start_idx = i * max_size
        end_idx = start_idx + sz.item()

        if end_idx > start_idx:
            slices.append(ag[start_idx:end_idx])

    ret = torch.cat(slices, dim=0)

    return ret.to(tensor)

#
#     def __init__(self, temperature=0.5, scale_by_temperature=True):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.scale_by_temperature = scale_by_temperature
#
#     def forward(self, features, labels=None, mask=None):
#         """
#         输入:
#             features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
#             labels: 每个样本的ground truth标签，尺寸是[batch_size].
#             mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1
#         输出:
#             loss值
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))
#         features = F.normalize(features, p=2, dim=1)
#         batch_size = features.shape[0]
#         # 关于labels参数
#         if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:  # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:  # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)
#         '''
#         示例:
#         labels:
#             tensor([[1.],
#                     [2.],
#                     [1.],
#                     [1.]])
#         mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
#             tensor([[1., 0., 1., 1.],
#                     [0., 1., 0., 0.],
#                     [1., 0., 1., 1.],
#                     [1., 0., 1., 1.]])
#         '''
#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features, features.T),
#             self.temperature)  # 计算两两样本间点乘相似度
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#         exp_logits = torch.exp(logits)
#         '''
#         logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
#         示例: logits: torch.size([4,4])
#         logits:
#             tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
#                     [-1.2576,  0.0000, -0.3367, -0.0725],
#                     [-1.3500, -0.1409, -0.1420,  0.0000],
#                     [-1.4312, -0.0776, -0.2009,  0.0000]])
#         '''
#         # 构建mask
#         logits_mask = torch.ones_like(mask) - torch.eye(batch_size)
#         positives_mask = mask * logits_mask
#         negatives_mask = 1. - mask
#         '''
#         但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
#         # 第ind行第ind位置填充为0
#         得到logits_mask:
#             tensor([[0., 1., 1., 1.],
#                     [1., 0., 1., 1.],
#                     [1., 1., 0., 1.],
#                     [1., 1., 1., 0.]])
#         positives_mask:
#         tensor([[0., 0., 1., 1.],
#                 [0., 0., 0., 0.],
#                 [1., 0., 0., 1.],
#                 [1., 0., 1., 0.]])
#         negatives_mask:
#         tensor([[0., 1., 0., 0.],
#                 [1., 0., 1., 1.],
#                 [0., 1., 0., 0.],
#                 [0., 1., 0., 0.]])
#         '''
#         num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
#         denominator = torch.sum(
#             exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
#             exp_logits * positives_mask, axis=1, keepdims=True)
#
#         log_probs = logits - torch.log(denominator)
#         if torch.any(torch.isnan(log_probs)):
#             raise ValueError("Log_prob has nan!")
#
#         log_probs = torch.sum(
#             log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
#                         num_positives_per_row > 0]
#         '''
#         计算正样本平均的log-likelihood
#         考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
#         所以这里只计算正样本个数>0的
#         '''
#         # loss
#         loss = -log_probs
#         if self.scale_by_temperature:
#             loss *= self.temperature
#         loss = loss.mean()
#         return loss