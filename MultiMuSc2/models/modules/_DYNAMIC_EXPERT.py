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
    representative_mode="historical_quality",
    admission_ttl_mode="historical_gap",
    candidate_mode="legacy",
    novel_ms_quantile=0.9,
    novel_support_quantile=0.3,
    novel_provisional_min=0.05,
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
      if representative_mode not in {
            "historical_quality",
            "latest_candidate",
        }:
            raise ValueError(
                "representative_mode must be historical_quality "
                "or latest_candidate"
            )
      if admission_ttl_mode not in {
            "historical_gap",
            "base",
        }:
            raise ValueError(
                "admission_ttl_mode must be historical_gap or base"
            )
      if candidate_mode not in {"legacy", "dual_path"}:
            raise ValueError("candidate_mode must be legacy or dual_path")
      if not 0.0 <= novel_ms_quantile <= 1.0:
            raise ValueError("novel_ms_quantile must be within [0, 1]")
      if not 0.0 <= novel_support_quantile <= 1.0:
            raise ValueError("novel_support_quantile must be within [0, 1]")
      if not 0.0 <= novel_provisional_min <= 1.0:
            raise ValueError("novel_provisional_min must be within [0, 1]")
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
      self.representative_mode = representative_mode
      self.admission_ttl_mode = admission_ttl_mode
      self.candidate_mode = candidate_mode
      self.novel_ms_quantile = float(novel_ms_quantile)
      self.novel_support_quantile = float(novel_support_quantile)
      self.novel_provisional_min = float(novel_provisional_min)
      
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

    def expert_audit_snapshot(self):
        return [
            {
                "expert_id": int(expert["expert_id"]),
                "image_id": int(expert["image_id"]),
                "cluster_id": int(expert["cluster_id"]),
                "admission_step": int(expert["admission_step"]),
                "cluster_support_at_admission": int(
                    expert["cluster_support_at_admission"]
                ),
                "max_support_gap": int(expert["max_support_gap"]),
                "last_supported_step": int(
                    expert["last_supported_step"]
                ),
                "support_count": int(expert["support_count"]),
                "patience": int(expert["patience"]),
                "ttl": int(expert["ttl"]),
                "deletion_step": expert["deletion_step"],
                "last_event": expert["last_event"],
                "last_channel_support": expert[
                    "last_channel_support"
                ],
                "representative_mode": expert[
                    "representative_mode"
                ],
                "admission_ttl_mode": expert[
                    "admission_ttl_mode"
                ],
                "admission_route": expert.get("admission_route", "legacy"),
            }
            for expert in self.active_experts
        ]

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

    def _novel_thresholds(self):
        novel_ms_threshold = (
            float(np.quantile(self.ms_history, self.novel_ms_quantile))
            if self.ms_history
            else None
        )
        novel_support_threshold = (
            float(
                np.quantile(
                    self.support_history,
                    self.novel_support_quantile,
                )
            )
            if self.support_history
            else None
        )
        return novel_ms_threshold, novel_support_threshold

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
        lifecycle_events = []

        for expert in self.active_experts:
            expert_id = expert["expert_id"]
            ttl_before = int(expert["ttl"])
            patience_before = int(expert["patience"])
            last_supported_step_before = int(
                expert["last_supported_step"]
            )
            support_gap = None

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

            signal_decision = event
            expert["last_event"] = event
            expert["last_channel_support"] = support

            deleted_this_step = False
            if expert["ttl"] <= 0:
                expert["ttl"] = 0
                expert["deletion_step"] = step
                expert["last_event"] = "deleted"
                deleted_this_step = True

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

            lifecycle_events.append(
                {
                    "step": int(step),
                    "expert_id": int(expert_id),
                    "expert_image_id": int(expert["image_id"]),
                    "cluster_id": int(expert["cluster_id"]),
                    "admission_step": int(expert["admission_step"]),
                    "age": int(step - expert["admission_step"]),
                    "channel_support": support,
                    "support_threshold": support_threshold,
                    "support_available": support is not None,
                    "threshold_available": support_threshold is not None,
                    "support_gap": support_gap,
                    "ttl_before": ttl_before,
                    "ttl_after": int(expert["ttl"]),
                    "patience_before": patience_before,
                    "patience_after": int(expert["patience"]),
                    "last_supported_step_before": (
                        last_supported_step_before
                    ),
                    "last_supported_step_after": int(
                        expert["last_supported_step"]
                    ),
                    "signal_decision": signal_decision,
                    "decision": expert["last_event"],
                    "refreshed": signal_decision == "ttl_refreshed",
                    "deleted_this_step": deleted_this_step,
                    "alive_after": not deleted_this_step,
                    "admission_candidate_image_id": None,
                    "admission_candidate_ms_score": None,
                    "admission_ms_threshold": None,
                    "admission_candidate_channel_support": None,
                    "admission_support_threshold": None,
                    "representative_mode": expert[
                        "representative_mode"
                    ],
                    "admission_ttl_mode": expert[
                        "admission_ttl_mode"
                    ],
                    "admission_route": expert.get(
                        "admission_route", "legacy"
                    ),
                }
            )

        self.active_experts = survivors
        return deleted_expert_ids, lifecycle_events


    def _consider_candidate(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        ms_threshold,
        support_threshold,
        provisional_channel_support,
        novel_ms_threshold,
        novel_support_threshold,
    ):
        legacy_candidate = bool(
            ms_score is not None
            and channel_support is not None
            and ms_threshold is not None
            and support_threshold is not None
            and ms_score <= ms_threshold
            and channel_support
            >= support_threshold
        )
        novel_candidate = bool(
            self.candidate_mode == "dual_path"
            and ms_score is not None
            and channel_support is not None
            and novel_ms_threshold is not None
            and novel_support_threshold is not None
            and provisional_channel_support is not None
            and ms_score >= novel_ms_threshold
            and channel_support <= novel_support_threshold
            and provisional_channel_support >= self.novel_provisional_min
        )
        candidate_route = (
            "legacy"
            if legacy_candidate
            else "new_distribution"
            if novel_candidate
            else None
        )

        if candidate_route is None:
            return False, None, None, None
        # 特征平均归一化
        embedding = self._image_embedding(
            dino_patch_features
        )

        if candidate_route == "legacy":
            quality = (
                channel_support
                / max(support_threshold, 1e-12)
                - ms_score
                / max(ms_threshold, 1e-12)
            )
        else:
            quality = float(provisional_channel_support)

        cluster_id = None

        if candidate_route == "legacy" and self.clusters:
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
                "candidate_route": candidate_route,
                "provisional_channel_support": provisional_channel_support,
            }
        )
        # 更新 cluster 中心
        cluster["embedding_sum"] += embedding
        # cluster能否进入专家
        if cluster["active_expert_id"] is not None:
            return True, cluster_id, None, candidate_route

        if (
            candidate_route == "legacy"
            and
            len(cluster["members"])
            < self.min_cluster_support
        ):
            return True, cluster_id, None, candidate_route

        if (
            len(self.active_experts)
            >= self.committee_cap
        ):
            return True, cluster_id, None, candidate_route

        # 选择代表
        if self.representative_mode == "historical_quality":
            representative = max(
                cluster["members"],
                key=lambda member: member["quality"],
            )
        else:
            representative = cluster["members"][-1]

        member_steps = sorted(
            member["step"]
            for member in cluster["members"]
        )

        historical_max_support_gap = max(
            (
                later - earlier
                for earlier, later in zip(
                    member_steps,
                    member_steps[1:],
                )
            ),
            default=0,
        )

        if (
            candidate_route == "legacy"
            and self.admission_ttl_mode == "historical_gap"
        ):
            max_support_gap = historical_max_support_gap
            patience = self._patience_from_gap(
                max_support_gap
            )
        else:
            max_support_gap = 0
            patience = self.base_ttl

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
            "representative_mode": self.representative_mode,
            "admission_ttl_mode": self.admission_ttl_mode,
            "admission_route": candidate_route,
        }

        cluster["active_expert_id"] = self.next_expert_id
        self.active_experts.append(expert)

        self.admitted_count += 1
        self.next_expert_id += 1

        return (
            True,
            cluster_id,
            expert["expert_id"],
            candidate_route,
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
        provisional_channel_support=None,
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
        provisional_channel_support = self._optional_score(
            "provisional_channel_support",
            provisional_channel_support,
            unit_interval=True,
        )

# 计算历史限制
        (
            ms_threshold,
            support_threshold,
        ) = self._historical_thresholds()
        (
            novel_ms_threshold,
            novel_support_threshold,
        ) = self._novel_thresholds()

        (
            deleted_expert_ids,
            expert_lifecycle_events,
        ) = (
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
            candidate_route,
        ) = self._consider_candidate(
            step,
            int(image_id),
            dino_patch_features,
            ms_score,
            channel_support,
            ms_threshold,
            support_threshold,
            provisional_channel_support,
            novel_ms_threshold,
            novel_support_threshold,
        )

        if admitted_expert_id is not None:
            admitted_expert = next(
                expert
                for expert in self.active_experts
                if expert["expert_id"] == admitted_expert_id
            )
            expert_lifecycle_events.append(
                {
                    "step": int(step),
                    "expert_id": int(admitted_expert_id),
                    "expert_image_id": int(
                        admitted_expert["image_id"]
                    ),
                    "cluster_id": int(
                        admitted_expert["cluster_id"]
                    ),
                    "admission_step": int(step),
                    "age": 0,
                    "channel_support": None,
                    "support_threshold": support_threshold,
                    "support_available": False,
                    "threshold_available": (
                        support_threshold is not None
                    ),
                    "support_gap": None,
                    "ttl_before": None,
                    "ttl_after": int(admitted_expert["ttl"]),
                    "patience_before": None,
                    "patience_after": int(
                        admitted_expert["patience"]
                    ),
                    "last_supported_step_before": None,
                    "last_supported_step_after": int(step),
                    "signal_decision": "admitted",
                    "decision": "admitted",
                    "refreshed": False,
                    "deleted_this_step": False,
                    "alive_after": True,
                    "admission_candidate_image_id": int(image_id),
                    "admission_candidate_ms_score": ms_score,
                    "admission_ms_threshold": ms_threshold,
                    "admission_candidate_channel_support": (
                        channel_support
                    ),
                    "admission_support_threshold": (
                        support_threshold
                    ),
                    "representative_mode": admitted_expert[
                        "representative_mode"
                    ],
                    "admission_ttl_mode": admitted_expert[
                        "admission_ttl_mode"
                    ],
                    "admission_route": admitted_expert[
                        "admission_route"
                    ],
                }
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
            "novel_support_threshold": novel_support_threshold,
            "novel_ms_threshold": novel_ms_threshold,
            "provisional_channel_support": provisional_channel_support,
            "is_candidate": is_candidate,
            "candidate_route": candidate_route,
            "cluster_id": cluster_id,
            "admitted_expert_id":
                admitted_expert_id,
            "deleted_expert_ids":
                deleted_expert_ids,
            "expert_lifecycle_events":
                expert_lifecycle_events,
            "active_expert_ids_after_step": [
                expert["expert_id"]
                for expert
                in self.active_experts
            ],
        }      

