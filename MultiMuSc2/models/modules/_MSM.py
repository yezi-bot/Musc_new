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


def safe_topmin_counts(reference_count, topmin_min=0, topmin_max=0.3):
    if reference_count < 1:
        raise ValueError("at least one reference image is required")

    k_max = (
        int(reference_count * topmin_max)
        if topmin_max < 1
        else int(topmin_max)
    )
    k_min = (
        int(reference_count * topmin_min)
        if topmin_min < 1
        else int(topmin_min)
    )
    k_max = max(1, min(reference_count, k_max))
    k_min = min(max(0, k_min), k_max - 1)
    return k_min, k_max, max(1, k_max - k_min)


def interval_average(distances, topmin_min=0, topmin_max=0.3):
    if distances.ndim != 2:
        raise ValueError("distances must have shape [patch_count, reference_count]")

    k_min, k_max, keep = safe_topmin_counts(
        distances.shape[1],
        topmin_min,
        topmin_max,
    )
    values = torch.topk(
        distances.float(),
        k_max,
        largest=False,
        sorted=True,
    ).values
    values = torch.topk(
        values,
        keep,
        largest=True,
        sorted=True,
    ).values
    return values.mean(dim=1)


def aggregate_reference_distances(
    query,
    references,
    topmin_min=0,
    topmin_max=0.3,
):
    if query.ndim != 2:
        raise ValueError("query must have shape [patch_count, feature_dim]")
    if references.ndim != 3:
        raise ValueError(
            "references must have shape [reference_count, patch_count, feature_dim]"
        )
    if references.shape[0] < 1:
        raise ValueError("at least one reference image is required")
    if query.shape[1] != references.shape[2]:
        raise ValueError("query and references must have the same feature dimension")

    patch_count = query.shape[0]
    reference_count, reference_patch_count, feature_dim = references.shape
    distances = torch.cdist(
        query.unsqueeze(0),
        references.reshape(-1, feature_dim),
    ).reshape(
        patch_count,
        reference_count,
        reference_patch_count,
    )
    nearest_patch = distances.min(dim=-1).values
    return interval_average(nearest_patch, topmin_min, topmin_max)


def MSM2_online(
    current_dino,
    current_clip,
    expert_dino,
    expert_clip,
    detect_fuser=None,
    fusion_mode="fuser",
    dino_scale=None,
    clip_scale=None,
    dino_weight=1.0,
    clip_weight=0.5,
    epsilon=1e-6,
    topmin_min=0,
    topmin_max=0.3,
):
    dino_distance = aggregate_reference_distances(
        current_dino,
        expert_dino,
        topmin_min,
        topmin_max,
    )
    if fusion_mode == "dino_only":
        return dino_distance, {
            "fusion_mode": fusion_mode,
            "reference_count": int(expert_dino.shape[0]),
            "dino_distance": dino_distance,
            "clip_distance": None,
            "fuser_score": None,
        }

    if current_clip is None or expert_clip is None:
        raise ValueError(f"{fusion_mode} fusion requires CLIP features")
    if expert_dino.shape[0] != expert_clip.shape[0]:
        raise ValueError("DINO and CLIP expert counts must match")

    clip_distance = aggregate_reference_distances(
        current_clip,
        expert_clip,
        topmin_min,
        topmin_max,
    ).to(dino_distance.device)

    if dino_distance.shape != clip_distance.shape:
        raise ValueError("DINO and CLIP patch counts must match")

    fuser_score = None
    if fusion_mode == "clip_only":
        patch_score = clip_distance
    elif fusion_mode == "fixed":
        if dino_scale is None or clip_scale is None:
            raise ValueError("fixed fusion requires historical DINO and CLIP scales")
        dino_scale = max(float(dino_scale), epsilon)
        clip_scale = max(float(clip_scale), epsilon)
        weight_sum = dino_weight + clip_weight
        if weight_sum <= 0:
            raise ValueError("fusion weights must have a positive sum")
        patch_score = (
            dino_weight * dino_distance / dino_scale
            + clip_weight * clip_distance / clip_scale
        ) / weight_sum
    elif fusion_mode == "fuser":
        if detect_fuser is None:
            raise ValueError("fuser fusion requires a fitted detect_fuser")
        distance_pairs = torch.stack(
            [dino_weight * dino_distance, clip_weight * clip_distance],
            dim=1,
        )
        fuser_values = detect_fuser.score_samples(
            distance_pairs.detach().cpu().numpy()
        )
        fuser_score = torch.as_tensor(
            fuser_values,
            device=dino_distance.device,
            dtype=dino_distance.dtype,
        )
        patch_score = fuser_score * dino_distance * clip_distance
    else:
        raise ValueError(f"unsupported fusion_mode: {fusion_mode}")

    audit = {
        "fusion_mode": fusion_mode,
        "reference_count": int(expert_dino.shape[0]),
        "dino_distance": dino_distance,
        "clip_distance": clip_distance,
        "fuser_score": fuser_score,
    }
    if fuser_score is not None:
        audit["fuser_min"] = float(fuser_score.min())
        audit["fuser_max"] = float(fuser_score.max())
        audit["fuser_mean"] = float(fuser_score.mean())
        audit["fuser_positive_ratio"] = float((fuser_score > 0).float().mean())
        audit["fuser_negative_ratio"] = float((fuser_score < 0).float().mean())
    return patch_score, audit


def build_causal_fuser_training_data(
    training_dino,
    training_clip,
    image_ids,
    member_steps,
    current_step,
    dino_weight=1.0,
    clip_weight=0.5,
    topmin_min=0,
    topmin_max=0.3,
):
    member_count = len(image_ids)
    if training_dino.ndim != 3 or training_clip.ndim != 3:
        raise ValueError("training features must have shape [image, patch, feature]")
    if not (
        training_dino.shape[0]
        == training_clip.shape[0]
        == member_count
        == len(member_steps)
    ):
        raise ValueError("DINO, CLIP, image IDs, and member steps must be aligned")
    if len(set(image_ids)) != member_count:
        raise ValueError("fuser training image IDs must be unique")
    if any(step >= current_step for step in member_steps):
        raise ValueError("fuser training members must be strictly historical")
    if member_count < 2:
        return {
            "available": False,
            "reason": "insufficient_history",
            "image_ids": list(image_ids),
            "train_pairs": None,
            "dino_scale": None,
            "clip_scale": None,
        }

    dino_distances = []
    clip_distances = []
    for index in range(member_count):
        reference_indices = [
            reference_index
            for reference_index in range(member_count)
            if reference_index != index
        ]
        dino_distance = aggregate_reference_distances(
            training_dino[index],
            training_dino[reference_indices],
            topmin_min,
            topmin_max,
        )
        clip_distance = aggregate_reference_distances(
            training_clip[index],
            training_clip[reference_indices],
            topmin_min,
            topmin_max,
        ).to(dino_distance.device)
        if dino_distance.shape != clip_distance.shape:
            raise ValueError("DINO and CLIP training patch counts must match")
        dino_distances.append(dino_distance)
        clip_distances.append(clip_distance)

    dino_history = torch.cat(dino_distances)
    clip_history = torch.cat(clip_distances)
    train_pairs = torch.stack(
        [dino_weight * dino_history, clip_weight * clip_history],
        dim=1,
    )
    if not torch.isfinite(train_pairs).all():
        raise ValueError("fuser training distances contain non-finite values")

    return {
        "available": True,
        "reason": None,
        "image_ids": list(image_ids),
        "train_pairs": train_pairs,
        "dino_scale": float(torch.quantile(dino_history.float(), 0.5)),
        "clip_scale": float(torch.quantile(clip_history.float(), 0.5)),
    }


def fit_causal_detect_fuser(
    detect_fuser,
    training_dino,
    training_clip,
    image_ids,
    member_steps,
    current_step,
    dino_weight=1.0,
    clip_weight=0.5,
    topmin_min=0,
    topmin_max=0.3,
):
    training = build_causal_fuser_training_data(
        training_dino,
        training_clip,
        image_ids,
        member_steps,
        current_step,
        dino_weight,
        clip_weight,
        topmin_min,
        topmin_max,
    )
    if not training["available"]:
        training["fitted"] = False
        return training

    train_pairs = training["train_pairs"]
    if torch.unique(train_pairs, dim=0).shape[0] < 2:
        training["available"] = False
        training["fitted"] = False
        training["reason"] = "degenerate_training_data"
        return training

    try:
        detect_fuser.fit(train_pairs.detach().cpu().numpy())
    except Exception as error:
        training["available"] = False
        training["fitted"] = False
        training["reason"] = f"fit_failed:{type(error).__name__}"
        return training

    training["fitted"] = True
    return training


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

    return interval_average(patch2image, topmin_min, topmin_max)

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

    patch2image11 = interval_average(
        patch2image,
        topmin_min,
        topmin_max,
    )
    patch2image = patch2image11.unsqueeze(0)
    patch2image22 = interval_average(
        patch2image2,
        topmin_min,
        topmin_max,
    )
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
    return interval_average(patch2image, topmin_min, topmin_max)
    
def compute_scores_slow2(Z,Z11,Z2,Z22, i, device,detect_fuser, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    patch2image2 = torch.tensor([]).to(device)
    for j in range(Z11.shape[0]):
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z11[j]), 1)[0].unsqueeze(1)), dim=1)
    for j2 in range(Z22.shape[0]):
            patch2image2 = torch.cat((patch2image2, torch.min(torch.cdist(Z2[i], Z22[j]), 1)[0].unsqueeze(1)), dim=1)
    patch2image11 = interval_average(
        patch2image,
        topmin_min,
        topmin_max,
    )
    patch2image = patch2image11.unsqueeze(0)
    patch2image22 = interval_average(
        patch2image2,
        topmin_min,
        topmin_max,
    )
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
