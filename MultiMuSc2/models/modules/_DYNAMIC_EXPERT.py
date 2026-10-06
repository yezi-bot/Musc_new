import math
import numpy as np
import torch
import torch.nn.functional as F


class DynamicExpertManager:
    def __init__(self,
    ms_quantile=0.3, 
    support_quantile=0.7,
    duplicate_similarity=0.98,
    min_cluster_support=2,
    committee_cap=5,
    base_ttl=5,
    max_ttl=20,
    ttl_gap_multiplier=2.0,
    ):
            # 历史异常的30％
      if not 0.0<=ms_quantile<=1.0:
            raise ValueError("ms_quantile must be within [0,1]")
            # 历史channel support的0.3
      if not 0.0 <= support_quantile <= 1.0:
            raise ValueError("support_quantile must be within [0, 1]")
            # DINO于cluster余弦相似度
      if not 0.0 <= duplicate_similarity <= 1.0:
            raise ValueError(
                "duplicate_similarity must be within [0, 1]"
            )
            # 一个cluster的确认必须包含两个相似候选图
      if min_cluster_support < 2:
            raise ValueError(
                "min_cluster_support must be at least 2"
            )
      if committee_cap < 1:
            raise ValueError("committee_cap must be at least 1")
      if base_ttl < 1:
            raise ValueError("base_ttl must be at least 1")
      if max_ttl < base_ttl:
            raise ValueError("max_ttl must be at least base_ttl")
      if ttl_gap_multiplier < 1.0:
            raise ValueError(
                "ttl_gap_multiplier must be at least 1"
            )  
      self.ms_quantile = float(ms_quantile)
      self.support_quantile = float(support_quantile)
      self.duplicate_similarity = float(
            duplicate_similarity
        )
      self.min_cluster_support = int(
            min_cluster_support
        )
      self.committee_cap = int(committee_cap)
      self.base_ttl = int(base_ttl)
      self.max_ttl = int(max_ttl)
      self.ttl_gap_multiplier = float(
            ttl_gap_multiplier
        )
      
      self.ms_history = []
      self.support_history = []
    #   候选聚类
      self.clusters = []
      self.active_experts = []
      self.retired_experts = []
      self.admitted_count = 0
      self.next_expert_id = 0
      self.last_step = -1

    def active_experts_before_step(self):
        return[
            {
                "expert_id":expert["expert_id"],
                "image_id":expert["image_id"],
                "cluster_id":expert["cluster_id"],
                "admission_step":expert["admission_step"],
            }
            for expert in self.active_experts
        ]

    def active_expert_before_step(self):
        return self.active_experts_before_step()

    def fuser_training_members(self, step):
        members = []
        active_cluster_ids = {
            expert["cluster_id"]
            for expert in self.active_experts
        }

        for cluster_id in active_cluster_ids:
            for member in self.clusters[cluster_id]["members"]:
                if member["step"] < step:
                    members.append(
                        {
                            "step": member["step"],
                            "image_id": member["image_id"],
                            "cluster_id": cluster_id,
                        }
                    )

        historical_members = []
        seen = set()
        for member in sorted(
            members,
            key=lambda item: (item["step"], item["image_id"]),
        ):
            image_id = member["image_id"]
            if image_id not in seen:
                seen.add(image_id)
                historical_members.append(member)
        return historical_members

    def fuser_training_image_ids(self, step):
        return [
            member["image_id"]
            for member in self.fuser_training_members(step)
        ]

# 根据相似候选出现间隔计算专家 TTL，限制在max和min之间
    def _patience_from_gap(self,support_gap):
        return min(
            self.max_ttl,
            max(
                self.base_ttl,
                int(math.ceil(  support_gap*self.ttl_gap_multiplier) ),
            ),
        )

    def _historical_thresholds(self):
        ms_threshold=(
            float(
                np.quantile(
                    self.ms_history,
                    self.ms_quantile,
                )
            )
            if self.ms_history
            else None
        )
        support_threshold=(
            float(
                # 计算分位数
                np.quantile(
                    self.support_history,
                    self.support_quantile,
                )
            )
            if self.support_history
            else None
        )
        return ms_threshold,support_threshold

    @staticmethod
    # 检查外部传入的分数是否合法
    def _optional_score(
        name,
        value,
        unit_interval=False,
    ):
     if value is None:
       return None
     value   =float(value)  
     if not np.isfinite(value):
        raise ValueError(
            f"{name} must be finite or None"
        )  

     if (
            unit_interval
            and not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"{name} must be within [0, 1]"
            )
     return value


    @staticmethod
    # [L,C]
    def _image_embedding(dino_patch_features):
        if dino_patch_features.ndim != 2:
            raise ValueError(
                "dino_patch_features must have shape "
                "[patch_count, feature_dim]"
            )

        if not torch.isfinite(
            dino_patch_features
        ).all():
            raise ValueError(
                "dino_patch_features contain "
                "non-finite values"
            )
        # 所有 patch 求平均:[C]
        embedding = (
            dino_patch_features
            .detach()
            .float()
            .mean(dim=0)
        )

        if float(embedding.norm()) == 0.0:
            raise ValueError(
                "mean DINO embedding must be non-zero"
            )
       # 归一化
        return F.normalize(
            embedding,
            dim=0,
        ).cpu()

    #管理存活专家
    def _advance_lifecycle(
        self,
        step,
        support_threshold,
        expert_channel_supports,
    ):
        deleted_expert_ids = []
        survivors = []

        for expert in self.active_experts:
            expert_id = expert["expert_id"]

            support = expert_channel_supports.get(
                expert_id
            )
            support = self._optional_score(
                f"expert_channel_supports[{expert_id}]",
                support,
                unit_interval=True,
            )

            if (
                support is None
                or support_threshold is None
            ):
                event = "signal_unavailable"

            elif support >= support_threshold:
                support_gap = (
                    step
                    - expert["last_supported_step"]
                )

                expert["max_support_gap"] = max(
                    expert["max_support_gap"],
                    support_gap,
                )

                expert["patience"] = (
                    self._patience_from_gap(
                        expert["max_support_gap"]
                    )
                )

                expert["ttl"] = expert["patience"]
                expert["last_supported_step"] = step
                expert["support_count"] += 1

                event = "ttl_refreshed"

            else:
                expert["ttl"] -= 1
                event = "ttl_decrement"

            expert["last_event"] = event
            expert["last_channel_support"] = support

            if expert["ttl"] <= 0:
                expert["ttl"] = 0
                expert["deletion_step"] = step
                expert["last_event"] = "deleted"

                cluster = self.clusters[
                    expert["cluster_id"]
                ]
                if (
                    cluster["active_expert_id"]
                    == expert_id
                ):
                    cluster["active_expert_id"] = None

                self.retired_experts.append(expert)
                deleted_expert_ids.append(expert_id)
            else:
                survivors.append(expert)

        self.active_experts = survivors
        return deleted_expert_ids


    def _consider_candidate(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        ms_threshold,
        support_threshold,
    ):
        is_candidate = bool(
            ms_score is not None
            and channel_support is not None
            and ms_threshold is not None
            and support_threshold is not None
            and ms_score <= ms_threshold
            and channel_support
            >= support_threshold
        )

        if not is_candidate:
            return False, None, None
        # 特征平均归一化
        embedding = self._image_embedding(
            dino_patch_features
        )
          
        # 候选质量  
        quality = (
            channel_support
            / max(support_threshold, 1e-12)
            - ms_score
            / max(ms_threshold, 1e-12)
        )

        cluster_id = None

        if self.clusters:
            centroids = torch.stack(
                [
                    F.normalize(
                        cluster["embedding_sum"],
                        dim=0,
                    )
                    for cluster in self.clusters
                ]
            )

            # 与当前所有图算相似度
            similarities = centroids @ embedding
            max_similarity, best_cluster = (
                similarities.max(dim=0)
            )

            #  大于0.98
            if (
                float(max_similarity)
                >= self.duplicate_similarity
            ):
                cluster_id = int(best_cluster)

      # 没有历史cluster即创建新cluster
        if cluster_id is None:
            cluster_id = len(self.clusters)

            self.clusters.append(
                {
                    "embedding_sum":
                        torch.zeros_like(embedding),
                    "members": [],
                    "active_expert_id": None,
                }
            )

        cluster = self.clusters[cluster_id]

        cluster["members"].append(
            {
                "step": step,
                "image_id": image_id,
                "embedding": embedding,
                "quality": float(quality),
                "ms_score": ms_score,
                "channel_support":
                    channel_support,
                "ms_threshold": ms_threshold,
                "support_threshold":
                    support_threshold,
            }
        )
        # 更新 cluster 中心
        cluster["embedding_sum"] += embedding
        # cluster能否进入专家
        if cluster["active_expert_id"] is not None:
            return True, cluster_id, None

        # 两张图片支持
        if (
            len(cluster["members"])
            < self.min_cluster_support
        ):
            return True, cluster_id, None

        if (
            len(self.active_experts)
            >= self.committee_cap
        ):
            return True, cluster_id, None

        # 选择代表
        representative = max(
            cluster["members"],
            key=lambda member: member["quality"],
        )

        member_steps = sorted(
            member["step"]
            for member in cluster["members"]
        )

        max_support_gap = max(
            later - earlier
            for earlier, later in zip(
                member_steps,
                member_steps[1:],
            )
        )

        # 计算ttl
        patience = self._patience_from_gap(
            max_support_gap
        )

        expert = {
            "expert_id": self.next_expert_id,
            "image_id":
                representative["image_id"],
            "cluster_id": cluster_id,
            "admission_step": step,
            "cluster_support_at_admission":
                len(cluster["members"]),
            "max_support_gap": max_support_gap,
            "last_supported_step": step,
            "support_count":
                len(cluster["members"]),
            "patience": patience,
            "ttl": patience,
            "deletion_step": None,
            "last_event": "admitted",
            "last_channel_support": None,
        }

        cluster["active_expert_id"] = self.next_expert_id
        self.active_experts.append(expert)

        self.admitted_count += 1
        self.next_expert_id += 1

        return (
            True,
            cluster_id,
            expert["expert_id"],
        )

# 每处理一张图时调用一次的总入口
    def advance(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        expert_channel_supports,
    ):
    # 连续性
        if step != self.last_step + 1:
            raise ValueError(
                f"step must be "
                f"{self.last_step + 1}, "
                f"got {step}"
            )

# 是否是字典
        if not isinstance(
            expert_channel_supports,
            dict,
        ):
            raise TypeError(
                "expert_channel_supports "
                "must be a dict"
            )

        ms_score = self._optional_score(
            "ms_score",
            ms_score,
        )

        channel_support = self._optional_score(
            "channel_support",
            channel_support,
            unit_interval=True,
        )

# 计算历史限制
        (
            ms_threshold,
            support_threshold,
        ) = self._historical_thresholds()

        deleted_expert_ids = (
            self._advance_lifecycle(
                step,
                support_threshold,
                expert_channel_supports,
            )
        )

# 专家的候选和准入
        (
            is_candidate,
            cluster_id,
            admitted_expert_id,
        ) = self._consider_candidate(
            step,
            int(image_id),
            dino_patch_features,
            ms_score,
            channel_support,
            ms_threshold,
            support_threshold,
        )

# 更新历史门限
        if ms_score is not None:
            self.ms_history.append(ms_score)

        if channel_support is not None:
            self.support_history.append(
                channel_support
            )

        self.last_step = step

        return {
            "step": step,
            "image_id": int(image_id),
            "ms_threshold": ms_threshold,
            "support_threshold":
                support_threshold,
            "is_candidate": is_candidate,
            "cluster_id": cluster_id,
            "admitted_expert_id":
                admitted_expert_id,
            "deleted_expert_ids":
                deleted_expert_ids,
            "active_expert_ids_after_step": [
                expert["expert_id"]
                for expert
                in self.active_experts
            ],
        }      

