import numpy as np
import torch
import pickle


def MMO(W, score, k_list=[1, 2, 3]):
    S_list = []
    # k 表示每张图只保留最相似的 k 个邻居
    for k in k_list:
        # 反向选取
        _, topk_matrix = torch.topk(W.float(), W.shape[0]-k, largest=False, sorted=True)
        W_mask = W.clone()
        for i in range(W.shape[0]):
            # 最不相似的 N-k 个"位置清零
            W_mask[i, topk_matrix[i]] = 0
        n = W.shape[-1]
        # 新建一个和 W 同形状的全零矩阵，后面只填对角线，作为度矩阵的逆
        D_ = torch.zeros_like(W).float()
        for i in range(n):
            # 对角线上填入每行相似度之和的倒数
            D_[i, i] = 1 / (W_mask[i,:].sum())
        P = D_ @ W_mask
        S = score.clone().unsqueeze(-1)
        S = P @ S
        S_list.append(S)
        # 多邻域规模融合
    S = torch.concat(S_list, -1).mean(-1)
    return S
# 构建相似度矩阵
def RsCIN(scores_old, cls_tokens=None, k_list=[0]):
    if cls_tokens is None or 0 in k_list:
        return scores_old
        # [N,C]
    cls_tokens = np.array(cls_tokens)
    scores = (scores_old - scores_old.min()) / (scores_old.max() - scores_old.min())
    # 计算相似度矩阵
    similarity_matrix = cls_tokens @ cls_tokens.T
    similarity_matrix = torch.tensor(similarity_matrix)
    scores_new = MMO(similarity_matrix.clone().float(), score=torch.tensor(scores).clone().float(), k_list=k_list)
    scores_new = scores_new.numpy()
    return scores_new
