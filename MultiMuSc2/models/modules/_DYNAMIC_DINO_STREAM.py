import math

import numpy as np
import torch
import torch.nn.functional as F

from ._CHANNEL import ChannelMemory
from ._DYNAMIC_EXPERT import DynamicExpertManager
from ._MSM import interval_average

# 尺度和层转成编号
def feature_key(r, layer):
    return f"r{int(r)}_l{int(layer)}"

# 分块计算DINO异常距离
def chunked_dino_score(
    query,
    references,
    device,
    topmin_min,
    topmin_max,
    chunk_size=8,
):
    if not references:
        return None

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

# 读取patch数和维度数
    query = query.detach().float().to(device)
    patch_count, feature_dim = query.shape
    patch_to_image = []

# 分批处理参考图
    for start in range(0, len(references), chunk_size):
        reference_chunk = torch.stack(
            references[start : start + chunk_size]
        ).float().to(device)

# 当前图所有 patch 到参考图所有 patch 的欧氏距离
        distances = torch.cdist(
            query.unsqueeze(0),
            reference_chunk.reshape(-1, feature_dim),
        ).reshape(
            patch_count,
            reference_chunk.shape[0],
            reference_chunk.shape[1],
        )

        patch_to_image.append(distances.amin(dim=-1))
        del reference_chunk, distances

    return interval_average(
        torch.cat(patch_to_image, dim=1),
        topmin_min=topmin_min,
        topmin_max=topmin_max,
    )

# 初始化在线状态
class DynamicDinoOnlineState:
    def __init__(
        self,
        device,
        image_size,
        feature_layers,
        r_list,
        committee_config,
        scoring_config,
    ):
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.feature_layers = [int(layer) for layer in feature_layers]
        self.r_list = [int(r) for r in r_list]

        self.committee_r = int(committee_config.get("r", 1))
        self.committee_layer = int(
            committee_config.get("dino_layer", 23)
        )
        if self.committee_r not in self.r_list:
            raise ValueError("committee r must exist in r_list")
        if self.committee_layer not in self.feature_layers:
            raise ValueError(
                "committee DINO layer must exist in feature_layers"
            )

        self.committee_key = feature_key(
            self.committee_r,
            self.committee_layer,
        )
        # channel距离阈值
        self.channel_quantile = float(
            committee_config.get(
                "channel_distance_quantile",
                0.7,
            )
        )
        # soft_reliability权重系数
        self.reliability_alpha = float(
            committee_config.get("reliability_alpha", 0.5)
        )

        self.topmin_min = float(
            scoring_config.get("topmin_min", 0.02)
        )
        self.topmin_max = float(
            scoring_config.get("topmin_max", 0.3)
        )
        # 读取 MSM 距离聚合区间和参考图分块数
        self.reference_chunk_size = int(
            scoring_config.get("reference_chunk_size", 8)
        )
# 建立channel
        self.memory = ChannelMemory(
            max_ttl=int(committee_config.get("channel_ttl", 5)),
            mature_span=float(
                committee_config.get("mature_span", 3.0)
            ),
            density_k=int(committee_config.get("density_k", 5)),
            device=self.device,
            position_radius=int(
                committee_config.get("position_radius", 1)
            ),
        )
# 建立专家管理器
        self.manager = DynamicExpertManager(
            ms_quantile=float(
                committee_config.get("ms_quantile", 0.3)
            ),
            support_quantile=float(
                committee_config.get("support_quantile", 0.7)
            ),
            duplicate_similarity=float(
                committee_config.get("duplicate_similarity", 0.98)
            ),
            min_cluster_support=int(
                committee_config.get("min_cluster_support", 2)
            ),
            committee_cap=int(
                committee_config.get("committee_cap", 5)
            ),
            base_ttl=int(
                committee_config.get("base_ttl", 5)
            ),
            max_ttl=int(
                committee_config.get("max_ttl", 20)
            ),
            ttl_gap_multiplier=float(
                committee_config.get("ttl_gap_multiplier", 2.0)
            ),
            representative_mode=committee_config.get(
                "representative_mode",
                "historical_quality",
            ),
            admission_ttl_mode=committee_config.get(
                "admission_ttl_mode",
                "historical_gap",
            ),
        )

        self.feature_bank = []
        self.distance_history = []
        self.timeline = []

    def _channel_support(
        self,
        features,
        grid_size,
        distance_threshold,
    ):
        distances = self.memory.patch_to_mature_distances(
            features,
            grid_size,
        )
        if distances is None:
            return None, None
        if distance_threshold is None:
            return None, distances

        support = float(
            (distances <= distance_threshold).float().mean()
        )
        return support, distances

    def _score_current(self, current_features, active_experts):
        step = len(self.feature_bank)

        if active_experts:
            reference_ids = [
                int(expert["image_id"])
                for expert in active_experts
            ]
            fallback_reason = None
            topmin_min = self.topmin_min
        elif step > 0:
            reference_ids = list(range(step))
            fallback_reason = "strict_history_dino"
            topmin_min = 0.0
        else:
            return None, None, [], "first_image_unavailable"

        if any(image_id >= step for image_id in reference_ids):
            raise RuntimeError(
                "online scoring attempted to use current or future images"
            )

        layer_scores = []
        # 历遍r和layer
        for r in self.r_list:
            for layer in self.feature_layers:
                key = feature_key(r, layer)
                references = [
                    self.feature_bank[image_id][key]
                    for image_id in reference_ids
                ]
                patch_score = chunked_dino_score(
                    query=current_features[key],
                    references=references,
                    device=self.device,
                    topmin_min=topmin_min,
                    topmin_max=self.topmin_max,
                    chunk_size=self.reference_chunk_size,
                )
                layer_scores.append(patch_score)
# 全部层和尺度平均
        patch_score = torch.stack(layer_scores).mean(dim=0)
        grid_size = math.isqrt(patch_score.numel())
        if grid_size * grid_size != patch_score.numel():
            raise ValueError("patch count must form a square grid")

        anomaly_map = F.interpolate(
            patch_score.reshape(1, 1, grid_size, grid_size),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=True,
        )[0, 0]

        image_score = float(anomaly_map.max().detach().cpu())
        return (
            anomaly_map.detach().cpu(),
            image_score,
            reference_ids,
            fallback_reason,
        )

    def process_features(self, current_features):
        step = len(self.feature_bank)
        expected_keys = {
            feature_key(r, layer)
            for r in self.r_list
            for layer in self.feature_layers
        }
        if set(current_features) != expected_keys:
            raise ValueError(
                "current_features do not match configured r/layer keys"
            )
# 开始时的专家
        active_before = self.manager.active_experts_before_step()
        expert_states_before = (
            self.manager.expert_audit_snapshot()
        )

        (
            anomaly_map,
            image_score,
            reference_ids,
            fallback_reason,
        ) = self._score_current(
            current_features,
            active_before,
        )

        committee_features = current_features[self.committee_key]
        grid_size = math.isqrt(committee_features.shape[0])
        if grid_size * grid_size != committee_features.shape[0]:
            raise ValueError(
                "committee patch count must form a square grid"
            )
# 构造历史委员会特征
        historical_committee = [
            features[self.committee_key]
            for features in self.feature_bank
        ]
        ms_patch_score = chunked_dino_score(
            query=committee_features,
            references=historical_committee,
            device=self.device,
            topmin_min=0.0,
            topmin_max=0.3,
            chunk_size=self.reference_chunk_size,
        )
        ms_score = (
            float(ms_patch_score.max().detach().cpu())
            if ms_patch_score is not None
            else None
        )
# 用累计距离计算channel阈值
        distance_threshold = (
            float(
                np.quantile(
                    self.distance_history,
                    self.channel_quantile,
                )
            )
            if self.distance_history
            else None
        )

        channel_support, current_distances = (
            self._channel_support(
                committee_features,
                grid_size,
                distance_threshold,
            )
        )
# 计算当前专家对历史channel的support
        expert_supports = {}
        for expert in active_before:
            expert_id = int(expert["expert_id"])
            expert_image_id = int(expert["image_id"])
            if expert_image_id >= step:
                raise RuntimeError(
                    "expert reference is not strictly historical"
                )

            expert_features = self.feature_bank[
                expert_image_id
            ][self.committee_key]
            support, _ = self._channel_support(
                expert_features,
                grid_size,
                distance_threshold,
            )
            expert_supports[expert_id] = support

        event = self.manager.advance(
            step=step,
            image_id=step,
            dino_patch_features=committee_features,
            ms_score=ms_score,
            channel_support=channel_support,
            expert_channel_supports=expert_supports,
        )
        expert_states_after = (
            self.manager.expert_audit_snapshot()
        )

        if current_distances is not None:
            finite = current_distances[
                torch.isfinite(current_distances)
            ]
            self.distance_history.extend(finite.tolist())

        _, reliability = self.memory.patch_reliability(
            committee_features,
            grid_size,
        )
        soft_reliability = (
            self.reliability_alpha
            + (1.0 - self.reliability_alpha) * reliability
        )

        self.memory.update(
            committee_features,
            image_id=step,
            grid_size=grid_size,
            patch_reliabilities=soft_reliability,
        )

        stored_features = {
            key: value.detach().cpu().clone()
            for key, value in current_features.items()
        }
        self.feature_bank.append(stored_features)

        record = {
            "step": step,
            "available": anomaly_map is not None,
            "image_score": image_score,
            "fallback_reason": fallback_reason,
            "scoring_reference_image_ids": reference_ids,
            "active_experts_before": active_before,
            "expert_states_before": expert_states_before,
            "ms_score": ms_score,
            "channel_distance_threshold": distance_threshold,
            "channel_support": channel_support,
            "expert_channel_supports": expert_supports,
            "admitted_expert_id": event["admitted_expert_id"],
            "deleted_expert_ids": event["deleted_expert_ids"],
            "expert_lifecycle_events": event[
                "expert_lifecycle_events"
            ],
            "active_experts_after":
                self.manager.active_experts_before_step(),
            "expert_states_after": expert_states_after,
            "channel_count_after": len(self.memory.channels),
            "mature_channel_count_after": len(
                self.memory.mature_channels()
            ),
            "ms_threshold": event["ms_threshold"],
            "support_threshold": event["support_threshold"],
            "is_candidate": event["is_candidate"],
            "candidate_cluster_id": event["cluster_id"],
        }
        self.timeline.append(record)

        return anomaly_map, record
