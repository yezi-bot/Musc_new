import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import linear_model
"""

We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""
def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0) #除测试图像外的其他所有测试图像[82,1369,1024]
    # 计算欧氏距离
    #当前照片，所有照片集合算距离，拆开，有image_num-1个照片
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)
    #求最小值，沿着最后一维（参照patch）
    patch2image = torch.min(patch2image, -1)[0]#[1369,11]

    # interval average1
    k_max = topmin_max
    k_min = topmin_min
    # 转化成整数
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    # 去掉最小的里面最小的剩下平均    
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def compute_scores_fast2(Z,Z11,Z2,Z22, i, device,detect_fuser, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    patch2image2 = torch.tensor([]).to(device)


    patch2image = torch.cdist(Z[i:i+1], Z11.reshape(-1, c)).reshape(patch_num, -1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]#[1369,11]

    patch2image2 = torch.cdist(Z2[i:i+1], Z22.reshape(-1, c)).reshape(patch_num, -1, patch_num)
    patch2image2 = torch.min(patch2image2, -1)[0]#[1369,11]

    # # interval average1
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    patch2image11 = torch.mean(patch2image, dim=1)  # [1369]
    patch2image = patch2image11.unsqueeze(0)
    # interval average2
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image2.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image2.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals2, _ = torch.topk(patch2image2.float(), k_max, largest=False, sorted=True)
    vals2, _ = torch.topk(vals2.float(), k_max-k_min, largest=True, sorted=True)

    patch2image2 = vals2.clone()
    patch2image22 = torch.mean(patch2image2, dim=1)  #[1369]
    patch2image2 =patch2image22.unsqueeze(0)
    s_map = torch.cat([1.0 * patch2image, 0.5 * patch2image2], dim=0).T
    s_map = s_map.cpu()
    s_map = torch.tensor(detect_fuser.score_samples(s_map))

    s_map = s_map.to(device)
    #anomaly_map = torch.cat((s_map.unsqueeze(0),patch2image22.unsqueeze(0),patch2image11.unsqueeze(0)),dim=0)
    # SVM分数，SVM分数*patch2image22*patch2image11
    return s_map*patch2image22*patch2image11
    #return torch.mean(anomaly_map, 0)

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)
    
def compute_scores_slow2(Z,Z11,Z2,Z22, i, device,detect_fuser, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    patch2image2 = torch.tensor([]).to(device)
    for j in range(Z11.shape[0]):
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z11[j]), 1)[0].unsqueeze(1)), dim=1)
    for j2 in range(Z22.shape[0]):
            patch2image2 = torch.cat((patch2image2, torch.min(torch.cdist(Z2[i], Z22[j]), 1)[0].unsqueeze(1)), dim=1)
    # # interval average1
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    patch2image11 = torch.mean(patch2image, dim=1)  # [1369]
    patch2image = patch2image11.unsqueeze(0)
    # interval average2
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image2.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image2.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals2, _ = torch.topk(patch2image2.float(), k_max, largest=False, sorted=True)
    vals2, _ = torch.topk(vals2.float(), k_max-k_min, largest=True, sorted=True)

    patch2image2 = vals2.clone()
    patch2image22 = torch.mean(patch2image2, dim=1)  #[1369]
    patch2image2 =patch2image22.unsqueeze(0)
    s_map = torch.cat([1.0 * patch2image, 0.5 * patch2image2], dim=0).T
    s_map = s_map.cpu()
    s_map = torch.tensor(detect_fuser.score_samples(s_map))
    s_map = s_map.to(device)
    anomaly_map = s_map*patch2image22*patch2image11
    return anomaly_map
def MSM(Z, device, topmin_min=0, topmin_max=0.3):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):
         #计算第i张异常得分数
        anomaly_scores_i = compute_scores_fast(Z, i, device, topmin_min, topmin_max).unsqueeze(0)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix

def MSM2(Z,Z11,Z2,Z22, device,detect_fuser, topmin_min=0, topmin_max=0.3):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):
    # for i in range(Z.shape[0]):
        anomaly_scores_i = compute_scores_fast2(Z,Z11,Z2,Z22, i, device,detect_fuser, topmin_min, topmin_max).unsqueeze(0)
        anomaly_scores_i = anomaly_scores_i.to(device)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)    # (N, B)
    return anomaly_scores_matrix

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
    