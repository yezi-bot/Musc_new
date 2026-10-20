import math
import itertools
import numpy as np
import torch
import torch.nn.functional as F


LAB_FEATURE_SCALES = np.asarray(
    [100.0, 100.0, 128.0, 255.0, 128.0, 255.0],
    dtype=np.float64,
)


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
    novel_admission_mode="immediate",
    novel_quarantine_steps=8,
    novel_quarantine_tail=4,
    novel_ratio_threshold=0.95,
    novel_provisional_weight=0.25,
    novel_confirmed_weight=0.5,
    novel_confirmed_ttl=16,
    novel_confirmed_members=1,
    novel_memory_window=16,
    novel_memory_keep_fraction=0.5,
    cold_start_mode="legacy",
    cold_start_min_samples=4,
    cold_start_max_samples=6,
    cold_start_required_members=3,
    cold_start_lab_enabled=False,
    lab_ewma_alpha=0.3,
    lab_ewma_threshold_k=3.0,
    lab_ewma_confirm_steps=2,
    photometric_candidate_enabled=False,
    photometric_quarantine_steps=4,
    photometric_prebuffer_size=2,
    photometric_support_confirm_steps=2,
    photometric_gate_mode="coverage_loss",
    photometric_candidate_window_mode="history",
    photometric_fixed_low_risk_members=4,
    photometric_dynamic_min_low_risk_members=3,
    photometric_dynamic_max_low_risk_members=6,
    photometric_episode_enabled=False,
    photometric_exit_confirm_steps=2,
    photometric_candidate_max_age=12,
    photometric_local_lab_distance_threshold=0.04,
    photometric_l_ewma_floor=0.04,
    photometric_ab_ewma_floor=0.01,
    photometric_consistency_fraction=0.75,
    photometric_min_low_risk_members=2,
    photometric_relevant_history_enabled=False,
    photometric_history_min=3,
    photometric_history_max=16,
    photometric_history_distance_threshold=3.0,
    photometric_dino_similarity_floor=0.9,
    photometric_budget_replacement=True,
    minimum_active_experts=0,
    drift_candidate_audit_enabled=False,
    drift_candidate_pool_size=4,
    drift_candidate_keep=3,
    drift_lab_noise_floor=0.01,
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
      if novel_admission_mode not in {
            "immediate",
            "quarantine",
            "provisional",
            "soft_confirmed",
            "channel_irbank",
            "channel_gate",
      }:
            raise ValueError(
                "novel_admission_mode must be immediate, quarantine, "
                "provisional, soft_confirmed, channel_irbank, or channel_gate"
            )
      if novel_quarantine_steps < 2:
            raise ValueError("novel_quarantine_steps must be at least 2")
      if not 1 <= novel_quarantine_tail <= novel_quarantine_steps:
            raise ValueError(
                "novel_quarantine_tail must be within quarantine steps"
            )
      if novel_ratio_threshold <= 0.0:
            raise ValueError("novel_ratio_threshold must be positive")
      if not 0.0 <= novel_provisional_weight <= 1.0:
            raise ValueError(
                "novel_provisional_weight must be within [0, 1]"
            )
      if not 0.0 <= novel_confirmed_weight <= 1.0:
            raise ValueError(
                "novel_confirmed_weight must be within [0, 1]"
            )
      if novel_confirmed_ttl < 1:
            raise ValueError("novel_confirmed_ttl must be at least 1")
      if novel_confirmed_members < 1:
            raise ValueError("novel_confirmed_members must be at least 1")
      if novel_confirmed_members > novel_quarantine_tail:
            raise ValueError(
                "novel_confirmed_members must not exceed quarantine tail"
            )
      if novel_memory_window < novel_quarantine_steps:
            raise ValueError(
                "novel_memory_window must be at least quarantine steps"
            )
      if not 0.0 < novel_memory_keep_fraction <= 1.0:
            raise ValueError(
                "novel_memory_keep_fraction must be within (0, 1]"
            )
      if cold_start_mode not in {"legacy", "adaptive"}:
            raise ValueError("cold_start_mode must be legacy or adaptive")
      if cold_start_min_samples < 2:
            raise ValueError("cold_start_min_samples must be at least 2")
      if cold_start_max_samples < cold_start_min_samples:
            raise ValueError(
                "cold_start_max_samples must be at least cold_start_min_samples"
            )
      if not 2 <= cold_start_required_members <= cold_start_min_samples:
            raise ValueError(
                "cold_start_required_members must be between 2 and "
                "cold_start_min_samples"
            )
      if not 0.0 < lab_ewma_alpha <= 1.0:
            raise ValueError("lab_ewma_alpha must be within (0, 1]")
      if lab_ewma_threshold_k < 0.0:
            raise ValueError("lab_ewma_threshold_k must be non-negative")
      if lab_ewma_confirm_steps < 1:
            raise ValueError("lab_ewma_confirm_steps must be positive")
      if photometric_quarantine_steps < 2:
            raise ValueError(
                "photometric_quarantine_steps must be at least 2"
            )
      if not 1 <= photometric_prebuffer_size <= photometric_quarantine_steps:
            raise ValueError(
                "photometric_prebuffer_size must be between 1 and "
                "photometric_quarantine_steps"
            )
      if photometric_support_confirm_steps < 1:
            raise ValueError(
                "photometric_support_confirm_steps must be positive"
            )
      if photometric_gate_mode not in {"coverage_loss", "lab_confirmed"}:
            raise ValueError(
                "photometric_gate_mode must be coverage_loss or lab_confirmed"
            )
      if photometric_candidate_window_mode not in {
            "history",
            "fixed_low_risk",
            "dynamic_low_risk",
      }:
            raise ValueError(
                "photometric_candidate_window_mode must be history, "
                "fixed_low_risk, or dynamic_low_risk"
            )
      if photometric_fixed_low_risk_members < 2:
            raise ValueError(
                "photometric_fixed_low_risk_members must be at least 2"
            )
      if photometric_dynamic_min_low_risk_members < 2:
            raise ValueError(
                "photometric_dynamic_min_low_risk_members must be at least 2"
            )
      if (
            photometric_dynamic_max_low_risk_members
            < photometric_dynamic_min_low_risk_members
      ):
            raise ValueError(
                "photometric_dynamic_max_low_risk_members must be at least "
                "photometric_dynamic_min_low_risk_members"
            )
      if photometric_exit_confirm_steps < 1:
            raise ValueError(
                "photometric_exit_confirm_steps must be positive"
            )
      if photometric_candidate_max_age < 1:
            raise ValueError(
                "photometric_candidate_max_age must be positive"
            )
      if photometric_local_lab_distance_threshold <= 0.0:
            raise ValueError(
                "photometric_local_lab_distance_threshold must be positive"
            )
      if photometric_l_ewma_floor <= 0.0:
            raise ValueError("photometric_l_ewma_floor must be positive")
      if photometric_ab_ewma_floor <= 0.0:
            raise ValueError("photometric_ab_ewma_floor must be positive")
      if not 0.0 < photometric_consistency_fraction <= 1.0:
            raise ValueError(
                "photometric_consistency_fraction must be within (0, 1]"
            )
      if not 1 <= photometric_min_low_risk_members <= photometric_quarantine_steps:
            raise ValueError(
                "photometric_min_low_risk_members must be between 1 and "
                "photometric_quarantine_steps"
            )
      if photometric_history_min < 2:
            raise ValueError("photometric_history_min must be at least 2")
      if photometric_history_max < photometric_history_min:
            raise ValueError(
                "photometric_history_max must be at least photometric_history_min"
            )
      if photometric_history_distance_threshold <= 0.0:
            raise ValueError(
                "photometric_history_distance_threshold must be positive"
            )
      if not 0.0 <= photometric_dino_similarity_floor <= 1.0:
            raise ValueError(
                "photometric_dino_similarity_floor must be within [0, 1]"
            )
      if not 0 <= minimum_active_experts <= committee_cap:
            raise ValueError(
                "minimum_active_experts must be between 0 and committee_cap"
            )
      if drift_candidate_pool_size < 2:
            raise ValueError("drift_candidate_pool_size must be at least 2")
      if not 2 <= drift_candidate_keep <= drift_candidate_pool_size:
            raise ValueError(
                "drift_candidate_keep must be between 2 and the pool size"
            )
      if drift_lab_noise_floor <= 0.0:
            raise ValueError("drift_lab_noise_floor must be positive")
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
      self.novel_admission_mode = novel_admission_mode
      self.novel_quarantine_steps = int(novel_quarantine_steps)
      self.novel_quarantine_tail = int(novel_quarantine_tail)
      self.novel_ratio_threshold = float(novel_ratio_threshold)
      self.novel_provisional_weight = float(novel_provisional_weight)
      self.novel_confirmed_weight = float(novel_confirmed_weight)
      self.novel_confirmed_ttl = int(novel_confirmed_ttl)
      self.novel_confirmed_members = int(novel_confirmed_members)
      self.novel_memory_window = int(novel_memory_window)
      self.novel_memory_keep_fraction = float(
            novel_memory_keep_fraction
        )
      self.cold_start_mode = cold_start_mode
      self.cold_start_min_samples = int(cold_start_min_samples)
      self.cold_start_max_samples = int(cold_start_max_samples)
      self.cold_start_required_members = int(cold_start_required_members)
      self.cold_start_lab_enabled = bool(cold_start_lab_enabled)
      self.lab_ewma_alpha = float(lab_ewma_alpha)
      self.lab_ewma_threshold_k = float(lab_ewma_threshold_k)
      self.lab_ewma_confirm_steps = int(lab_ewma_confirm_steps)
      self.photometric_candidate_enabled = bool(
            photometric_candidate_enabled
        )
      self.photometric_quarantine_steps = int(
            photometric_quarantine_steps
        )
      self.photometric_prebuffer_size = int(
            photometric_prebuffer_size
        )
      self.photometric_support_confirm_steps = int(
            photometric_support_confirm_steps
        )
      self.photometric_gate_mode = photometric_gate_mode
      self.photometric_candidate_window_mode = (
            photometric_candidate_window_mode
      )
      self.photometric_fixed_low_risk_members = int(
            photometric_fixed_low_risk_members
      )
      self.photometric_dynamic_min_low_risk_members = int(
            photometric_dynamic_min_low_risk_members
      )
      self.photometric_dynamic_max_low_risk_members = int(
            photometric_dynamic_max_low_risk_members
      )
      self.photometric_episode_enabled = bool(
            photometric_episode_enabled
      )
      self.photometric_exit_confirm_steps = int(
            photometric_exit_confirm_steps
      )
      self.photometric_candidate_max_age = int(
            photometric_candidate_max_age
      )
      self.photometric_local_lab_distance_threshold = float(
            photometric_local_lab_distance_threshold
      )
      self.photometric_l_ewma_floor = float(
            photometric_l_ewma_floor
        )
      self.photometric_ab_ewma_floor = float(
            photometric_ab_ewma_floor
        )
      self.photometric_consistency_fraction = float(
            photometric_consistency_fraction
        )
      self.photometric_min_low_risk_members = int(
            photometric_min_low_risk_members
        )
      self.photometric_relevant_history_enabled = bool(
            photometric_relevant_history_enabled
      )
      self.photometric_history_min = int(photometric_history_min)
      self.photometric_history_max = int(photometric_history_max)
      self.photometric_history_distance_threshold = float(
            photometric_history_distance_threshold
      )
      self.photometric_dino_similarity_floor = float(
            photometric_dino_similarity_floor
      )
      self.photometric_budget_replacement = bool(
            photometric_budget_replacement
      )
      self.minimum_active_experts = int(minimum_active_experts)
      self.drift_candidate_audit_enabled = bool(
            drift_candidate_audit_enabled
        )
      self.drift_candidate_pool_size = int(drift_candidate_pool_size)
      self.drift_candidate_keep = int(drift_candidate_keep)
      self.drift_lab_noise_floor = float(drift_lab_noise_floor)
      
      self.ms_history = []
      self.support_history = []
    #   候选聚类
      self.clusters = []
      self.active_experts = []
      self.retired_experts = []
      self.admitted_count = 0
      self.next_expert_id = 0
      self.last_step = -1
      self.novel_quarantine = None
      self.novel_quarantine_route = None
      self.last_novel_quarantine_event = None
      self.confirmed_provisional = None
      self.bootstrap_members = []
      self.last_bootstrap_event = None
      self.drift_candidate_pool = None
      self.last_drift_candidate_audit_event = None
      self.photometric_prebuffer = []
      self.photometric_armed_expert_id = None
      self.photometric_support_streak = 0
      self.photometric_trigger_expert_id = None
      self.photometric_trigger_lab_center = None
      self.last_photometric_gate = None
      self.photometric_episode_active = False
      self.photometric_episode_pending = False
      self.photometric_episode_reference_expert_id = None
      self.photometric_episode_lab_center = None
      self.photometric_episode_start_step = None
      self.photometric_recovery_streak = 0
      self.photometric_pool_abort_reason = None
      self.last_budget_replacement = None

    def provisional_expert_before_step(self):
        if (
            self.novel_admission_mode
            not in {
                "provisional",
                "soft_confirmed",
                "channel_irbank",
                "channel_gate",
            }
        ):
            return None
        if self.novel_quarantine:
            if self.novel_quarantine_route == "photometric_drift":
                return None
            if self.novel_admission_mode in {
                "channel_irbank",
                "channel_gate",
            }:
                representatives = self._select_irbank_members(
                    self.novel_quarantine
                )
                if not representatives:
                    return None
                representative = representatives[0]
                state = (
                    "collecting_channel_gate"
                    if self.novel_admission_mode == "channel_gate"
                    else "collecting_irbank"
                )
            else:
                representative = min(
                    self.novel_quarantine[-self.novel_quarantine_tail:],
                    key=lambda item: item["ms_short_score"],
                )
                representatives = [representative]
                state = "collecting"
            start_step = self.novel_quarantine[0]["step"]
            pool_size = len(self.novel_quarantine)
            weight = self.novel_provisional_weight
        elif self.confirmed_provisional is not None:
            representatives = self.confirmed_provisional["members"]
            representative = representatives[0]
            start_step = self.confirmed_provisional["start_step"]
            pool_size = self.novel_quarantine_steps
            weight = self.novel_confirmed_weight
            state = (
                "confirmed_irbank"
                if self.novel_admission_mode == "channel_irbank"
                else "confirmed"
            )
        else:
            return None
        return {
            "image_id": int(representative["image_id"]),
            "image_ids": [
                int(member["image_id"])
                for member in (
                    representatives
                    if state in {"confirmed", "confirmed_irbank"}
                    or self.novel_admission_mode
                    in {"channel_irbank", "channel_gate"}
                    else [representative]
                )
            ],
            "memory_members": [
                {
                    "image_id": int(member["image_id"]),
                    "patch_ids": list(member.get("memory_patch_ids", [])),
                }
                for member in representatives
            ],
            "representative_step": int(representative["step"]),
            "start_step": int(start_step),
            "pool_size": int(pool_size),
            "weight": float(weight),
            "state": state,
        }

    def active_experts_before_step(self):
        snapshots = []
        for expert in self.active_experts:
            snapshot = {
                "expert_id":expert["expert_id"],
                "image_id":expert["image_id"],
                "cluster_id":expert["cluster_id"],
                "admission_step":expert["admission_step"],
            }
            if expert.get("admission_route", "legacy") != "legacy":
                snapshot["admission_route"] = expert["admission_route"]
            if expert.get("lab_state") is not None:
                snapshot["lab_state"] = dict(expert["lab_state"])
            snapshots.append(snapshot)
        return snapshots

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

    @staticmethod
    def _lab_feature(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (6,):
            raise ValueError("lab_feature must have shape [6]")
        if not np.isfinite(value).all():
            raise ValueError("lab_feature must be finite")
        return value

    @staticmethod
    def _lab_distances(feature, center):
        delta = (feature - center) / LAB_FEATURE_SCALES
        return {
            "l_distance": float(np.linalg.norm(delta[:2])),
            "ab_distance": float(np.linalg.norm(delta[2:])),
            "total_distance": float(np.linalg.norm(delta)),
        }

    def _build_lab_state(self, members):
        features = [
            member["lab_feature"]
            for member in members
            if member["lab_feature"] is not None
        ]
        if not self.cold_start_lab_enabled or len(features) != len(members):
            return None

        features = np.stack(features, axis=0)
        center = np.median(features, axis=0)
        profiles = [
            self._lab_distances(feature, center)
            for feature in features
        ]
        state = {
            "member_image_ids": [int(member["image_id"]) for member in members],
            "center": center.tolist(),
            "recent_lab_history": [
                {
                    "step": int(member["step"]),
                    "image_id": int(member["image_id"]),
                    "feature": np.asarray(
                        member["lab_feature"], dtype=np.float64
                    ).tolist(),
                }
                for member in members
            ],
            "alpha": self.lab_ewma_alpha,
            "threshold_k": self.lab_ewma_threshold_k,
            "confirm_steps": self.lab_ewma_confirm_steps,
            "l_streak": 0,
            "ab_streak": 0,
            "drift_confirmed": False,
        }
        for channel in ("l", "ab"):
            values = np.asarray(
                [profile[f"{channel}_distance"] for profile in profiles],
                dtype=np.float64,
            )
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = max(1.4826 * mad, 1e-12)
            state[f"{channel}_baseline_median"] = median
            state[f"{channel}_baseline_mad"] = mad
            state[f"{channel}_threshold"] = float(
                median + self.lab_ewma_threshold_k * scale
            )
            state[f"{channel}_ewma"] = median
        return state

    def _relevant_lab_history_profile(self, state, current_record):
        records = list(state.get("recent_lab_history", []))
        records.append(current_record)
        records = records[-self.photometric_history_max:]
        if len(records) < self.photometric_history_min:
            return {
                "relevant_history_compatible": False,
                "relevant_history_size": len(records),
                "relevant_history_available": len(records),
                "relevant_image_ids": [
                    int(record["image_id"]) for record in records
                ],
                "relevant_steps": [int(record["step"]) for record in records],
                "relevant_distance": None,
                "history_shortened": False,
            }

        normalized = np.stack(
            [
                np.asarray(record["feature"], dtype=np.float64)
                / LAB_FEATURE_SCALES
                for record in records
            ],
            axis=0,
        )
        current = normalized[-1]
        best = None
        max_size = min(len(records), self.photometric_history_max)
        for size in range(self.photometric_history_min, max_size + 1):
            window = normalized[-size:]
            center = np.median(window, axis=0)
            mad = np.median(np.abs(window - center), axis=0)
            scale = np.maximum(
                1.4826 * mad,
                self.drift_lab_noise_floor,
            )
            distance = float(
                np.linalg.norm((current - center) / scale)
                / math.sqrt(current.shape[0])
            )
            if distance > self.photometric_history_distance_threshold:
                break
            best = {
                "size": size,
                "center": center,
                "variance": np.square(scale),
                "distance": distance,
            }

        if best is None:
            selected_records = records[-self.photometric_history_min:]
            return {
                "relevant_history_compatible": False,
                "relevant_history_size": len(selected_records),
                "relevant_history_available": len(records),
                "relevant_image_ids": [
                    int(record["image_id"]) for record in selected_records
                ],
                "relevant_steps": [
                    int(record["step"]) for record in selected_records
                ],
                "relevant_distance": None,
                "history_shortened": len(selected_records) < len(records),
            }

        selected_records = records[-best["size"]:]
        relevant_center = best["center"] * LAB_FEATURE_SCALES
        separation = self._lab_distances(
            relevant_center,
            np.asarray(state["center"], dtype=np.float64),
        )
        return {
            "relevant_history_compatible": True,
            "relevant_history_size": int(best["size"]),
            "relevant_history_available": len(records),
            "relevant_image_ids": [
                int(record["image_id"]) for record in selected_records
            ],
            "relevant_steps": [
                int(record["step"]) for record in selected_records
            ],
            "relevant_distance": float(best["distance"]),
            "relevant_center": relevant_center.tolist(),
            "relevant_variance_diag": best["variance"].tolist(),
            "relevant_l_separation": separation["l_distance"],
            "relevant_ab_separation": separation["ab_distance"],
            "relevant_total_separation": separation["total_distance"],
            "history_shortened": int(best["size"]) < len(records),
        }

    def _advance_cold_start(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        lab_feature,
    ):
        embedding = self._image_embedding(dino_patch_features)
        member = {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": embedding,
            "quality": float(channel_support or 0.0) - float(ms_score or 0.0),
            "ms_score": ms_score,
            "channel_support": channel_support,
            "ms_threshold": None,
            "support_threshold": None,
            "candidate_route": "bootstrap",
            "provisional_channel_support": None,
            "lab_feature": self._lab_feature(lab_feature),
        }
        self.bootstrap_members.append(member)
        pool_size = len(self.bootstrap_members)
        self.last_bootstrap_event = {
            "decision": "collecting",
            "pool_size": pool_size,
            "selected_image_ids": [],
            "forced": False,
        }
        if pool_size < self.cold_start_min_samples:
            return True, None, None, "bootstrap"

        combinations = list(
            itertools.combinations(
                range(pool_size),
                self.cold_start_required_members,
            )
        )
        best = None
        for indices in combinations:
            selected = [self.bootstrap_members[index] for index in indices]
            embeddings = torch.stack(
                [item["embedding"] for item in selected]
            )
            similarities = embeddings @ embeddings.T
            pair_values = similarities[
                torch.triu_indices(
                    len(selected),
                    len(selected),
                    offset=1,
                ).unbind()
            ]
            minimum = float(pair_values.min())
            mean = float(pair_values.mean())
            score = (minimum, mean, tuple(-index for index in indices))
            if best is None or score > best[0]:
                best = (score, indices, selected, similarities)

        minimum_similarity = best[0][0]
        forced = pool_size >= self.cold_start_max_samples
        if minimum_similarity < self.duplicate_similarity and not forced:
            return True, None, None, "bootstrap"

        _, _, selected, similarities = best
        medoid_index = int(similarities.mean(dim=1).argmax())
        representative = selected[medoid_index]
        cluster_id = len(self.clusters)
        self.clusters.append(
            {
                "embedding_sum": torch.stack(
                    [item["embedding"] for item in selected]
                ).sum(dim=0),
                "members": selected,
                "active_expert_id": self.next_expert_id,
            }
        )
        expert = {
            "expert_id": self.next_expert_id,
            "image_id": representative["image_id"],
            "cluster_id": cluster_id,
            "admission_step": int(step),
            "cluster_support_at_admission": len(selected),
            "max_support_gap": 0,
            "last_supported_step": int(step),
            "support_count": len(selected),
            "patience": self.base_ttl,
            "ttl": self.base_ttl,
            "deletion_step": None,
            "last_event": "admitted",
            "last_channel_support": None,
            "representative_mode": "bootstrap_medoid",
            "admission_ttl_mode": "base",
            "admission_route": "bootstrap",
            "lab_state": self._build_lab_state(selected),
        }
        self.active_experts.append(expert)
        self.admitted_count += 1
        self.next_expert_id += 1
        self.last_bootstrap_event = {
            "decision": "admitted",
            "pool_size": pool_size,
            "selected_image_ids": [
                int(item["image_id"]) for item in selected
            ],
            "representative_image_id": int(representative["image_id"]),
            "minimum_similarity": minimum_similarity,
            "forced": bool(forced),
        }
        return True, cluster_id, expert["expert_id"], "bootstrap"

    def advance_lab_state(self, lab_feature, step=None, image_id=None):
        feature = self._lab_feature(lab_feature)
        if feature is None:
            return None
        candidates = [
            expert
            for expert in self.active_experts
            if expert.get("lab_state") is not None
        ]
        if not candidates:
            return None

        profiles = []
        for expert in candidates:
            state = expert["lab_state"]
            distances = self._lab_distances(
                feature,
                np.asarray(state["center"], dtype=np.float64),
            )
            profiles.append((distances["total_distance"], expert, distances))
        _, expert, distances = min(profiles, key=lambda item: item[0])
        state = expert["lab_state"]
        relevant_profile = None
        if self.photometric_relevant_history_enabled:
            record_step = self.last_step + 1 if step is None else int(step)
            record_image_id = record_step if image_id is None else int(image_id)
            current_record = {
                "step": record_step,
                "image_id": record_image_id,
                "feature": feature.tolist(),
            }
            relevant_profile = self._relevant_lab_history_profile(
                state,
                current_record,
            )
            history = list(state.get("recent_lab_history", []))
            history.append(current_record)
            state["recent_lab_history"] = history[-self.photometric_history_max:]
        for channel in ("l", "ab"):
            state[f"{channel}_ewma"] = float(
                state["alpha"] * distances[f"{channel}_distance"]
                + (1.0 - state["alpha"]) * state[f"{channel}_ewma"]
            )
            hit = state[f"{channel}_ewma"] > state[f"{channel}_threshold"]
            state[f"{channel}_streak"] = (
                state[f"{channel}_streak"] + 1 if hit else 0
            )
        previously_confirmed = bool(state["drift_confirmed"])
        state["drift_confirmed"] = bool(
            state["l_streak"] >= state["confirm_steps"]
            or state["ab_streak"] >= state["confirm_steps"]
        )
        just_confirmed = bool(
            state["drift_confirmed"] and not previously_confirmed
        )
        return {
            "expert_id": int(expert["expert_id"]),
            "lab_center": np.asarray(
                state["center"],
                dtype=np.float64,
            ).copy().tolist(),
            **distances,
            "l_ewma": state["l_ewma"],
            "ab_ewma": state["ab_ewma"],
            "l_threshold": state["l_threshold"],
            "ab_threshold": state["ab_threshold"],
            "l_streak": int(state["l_streak"]),
            "ab_streak": int(state["ab_streak"]),
            "drift_confirmed": state["drift_confirmed"],
            "just_confirmed": just_confirmed,
            **(relevant_profile or {}),
        }

    def _advance_photometric_gate(
        self,
        lab_drift_profile,
        support_threshold,
        expert_channel_supports,
        step=None,
    ):
        self.last_photometric_gate = None
        self.photometric_pool_abort_reason = None
        if not self.photometric_candidate_enabled:
            self.photometric_armed_expert_id = None
            self.photometric_support_streak = 0
            return False, None

        pool_active = bool(
            self.novel_quarantine is not None
            and self.novel_quarantine_route == "photometric_drift"
        )
        if lab_drift_profile is None:
            episode_ended = False
            ended_reference_expert_id = None
            ended_start_step = None
            reference_active = any(
                int(expert["expert_id"])
                == self.photometric_episode_reference_expert_id
                for expert in self.active_experts
            )
            if (
                self.photometric_episode_enabled
                and self.photometric_episode_active
                and not reference_active
            ):
                ended_reference_expert_id = (
                    self.photometric_episode_reference_expert_id
                )
                ended_start_step = self.photometric_episode_start_step
                if pool_active:
                    self.photometric_pool_abort_reason = (
                        "aborted_photometric_reference_missing"
                    )
                self._reset_photometric_episode()
                episode_ended = True
            elif (
                self.photometric_episode_enabled
                and pool_active
                and step is not None
                and int(step) - int(self.novel_quarantine[0]["step"])
                >= self.photometric_candidate_max_age
            ):
                self.photometric_pool_abort_reason = (
                    "aborted_photometric_timeout"
                )
            self.last_photometric_gate = {
                "expert_id": None,
                "robust_shift": False,
                "pool_active": pool_active,
                "triggered": False,
                "episode_enabled": self.photometric_episode_enabled,
                "episode_active": self.photometric_episode_active,
                "episode_pending": self.photometric_episode_pending,
                "episode_started": False,
                "episode_ended": episode_ended,
                "ended_episode_reference_expert_id": (
                    ended_reference_expert_id
                ),
                "ended_episode_start_step": ended_start_step,
                "recovery_streak": int(self.photometric_recovery_streak),
                "pool_abort_reason": self.photometric_pool_abort_reason,
            }
            return False, self.photometric_episode_reference_expert_id

        expert_id = int(lab_drift_profile["expert_id"])
        robust_l = float(lab_drift_profile["l_ewma"]) >= max(
            float(lab_drift_profile["l_threshold"]),
            self.photometric_l_ewma_floor,
        )
        robust_ab = float(lab_drift_profile["ab_ewma"]) >= max(
            float(lab_drift_profile["ab_threshold"]),
            self.photometric_ab_ewma_floor,
        )
        robust_shift = bool(
            lab_drift_profile.get("drift_confirmed", False)
            and (robust_l or robust_ab)
        )
        if self.photometric_relevant_history_enabled:
            relevant_l = float(
                lab_drift_profile.get("relevant_l_separation", 0.0)
            ) >= self.photometric_l_ewma_floor
            relevant_ab = float(
                lab_drift_profile.get("relevant_ab_separation", 0.0)
            ) >= self.photometric_ab_ewma_floor
            robust_shift = bool(
                robust_shift
                and lab_drift_profile.get(
                    "relevant_history_compatible", False
                )
                and (relevant_l or relevant_ab)
            )
        else:
            relevant_l = None
            relevant_ab = None
        support = self._optional_score(
            f"expert_channel_supports[{expert_id}]",
            expert_channel_supports.get(expert_id),
            unit_interval=True,
        )
        coverage_loss = bool(
            robust_shift
            and support is not None
            and support_threshold is not None
            and support <= support_threshold
        )

        if not robust_shift:
            self.photometric_armed_expert_id = None
            self.photometric_support_streak = 0
        else:
            if self.photometric_armed_expert_id != expert_id:
                self.photometric_armed_expert_id = expert_id
                self.photometric_support_streak = 0
            self.photometric_support_streak = (
                self.photometric_support_streak + 1
                if coverage_loss
                else 0
            )

        gate_hit = bool(
            robust_shift
            and not pool_active
            and (
                self.photometric_gate_mode == "lab_confirmed"
                or self.photometric_support_streak
                >= self.photometric_support_confirm_steps
            )
        )
        episode_started = False
        episode_ended = False
        ended_reference_expert_id = None
        ended_start_step = None
        recovery_streak_at_end = None
        if self.photometric_episode_enabled:
            if not self.photometric_episode_active and gate_hit:
                self.photometric_episode_active = True
                self.photometric_episode_pending = True
                self.photometric_episode_reference_expert_id = expert_id
                self.photometric_episode_lab_center = np.asarray(
                    lab_drift_profile["lab_center"],
                    dtype=np.float64,
                ).copy()
                self.photometric_episode_start_step = (
                    int(step) if step is not None else None
                )
                self.photometric_recovery_streak = 0
                episode_started = True
            elif self.photometric_episode_active:
                self.photometric_recovery_streak = (
                    0
                    if robust_shift
                    else self.photometric_recovery_streak + 1
                )

                reference_active = any(
                    int(expert["expert_id"])
                    == self.photometric_episode_reference_expert_id
                    for expert in self.active_experts
                )
                recovered = (
                    self.photometric_recovery_streak
                    >= self.photometric_exit_confirm_steps
                )
                if not reference_active:
                    ended_reference_expert_id = (
                        self.photometric_episode_reference_expert_id
                    )
                    ended_start_step = self.photometric_episode_start_step
                    recovery_streak_at_end = int(
                        self.photometric_recovery_streak
                    )
                    if pool_active:
                        self.photometric_pool_abort_reason = (
                            "aborted_photometric_reference_missing"
                        )
                    self._reset_photometric_episode()
                    episode_ended = True
                elif recovered:
                    ended_reference_expert_id = (
                        self.photometric_episode_reference_expert_id
                    )
                    ended_start_step = self.photometric_episode_start_step
                    recovery_streak_at_end = int(
                        self.photometric_recovery_streak
                    )
                    if pool_active:
                        self.photometric_pool_abort_reason = (
                            "aborted_photometric_recovered"
                        )
                    self._reset_photometric_episode()
                    episode_ended = True

            if (
                pool_active
                and self.photometric_pool_abort_reason is None
                and step is not None
                and int(step) - int(self.novel_quarantine[0]["step"])
                >= self.photometric_candidate_max_age
            ):
                self.photometric_pool_abort_reason = (
                    "aborted_photometric_timeout"
                )
            triggered = episode_started
        else:
            triggered = gate_hit
        self.last_photometric_gate = {
            "expert_id": expert_id,
            "robust_l": robust_l,
            "robust_ab": robust_ab,
            "robust_shift": robust_shift,
            "relevant_l": relevant_l,
            "relevant_ab": relevant_ab,
            "relevant_history_size": lab_drift_profile.get(
                "relevant_history_size"
            ),
            "relevant_history_available": lab_drift_profile.get(
                "relevant_history_available"
            ),
            "history_shortened": lab_drift_profile.get(
                "history_shortened"
            ),
            "expert_channel_support": support,
            "support_threshold": support_threshold,
            "coverage_loss": coverage_loss,
            "support_streak": int(self.photometric_support_streak),
            "required_support_steps": self.photometric_support_confirm_steps,
            "gate_mode": self.photometric_gate_mode,
            "pool_active": pool_active,
            "triggered": triggered,
            "episode_enabled": self.photometric_episode_enabled,
            "episode_active": self.photometric_episode_active,
            "episode_pending": self.photometric_episode_pending,
            "episode_started": episode_started,
            "episode_ended": episode_ended,
            "episode_reference_expert_id": (
                self.photometric_episode_reference_expert_id
            ),
            "episode_start_step": self.photometric_episode_start_step,
            "ended_episode_reference_expert_id": (
                ended_reference_expert_id
            ),
            "ended_episode_start_step": ended_start_step,
            "recovery_streak_at_end": recovery_streak_at_end,
            "recovery_streak": int(self.photometric_recovery_streak),
            "required_recovery_steps": self.photometric_exit_confirm_steps,
            "pool_abort_reason": self.photometric_pool_abort_reason,
        }
        return (
            triggered,
            self.photometric_episode_reference_expert_id
            if self.photometric_episode_enabled
            and self.photometric_episode_active
            else expert_id,
        )

    def _reset_photometric_episode(self):
        self.photometric_episode_active = False
        self.photometric_episode_pending = False
        self.photometric_episode_reference_expert_id = None
        self.photometric_episode_lab_center = None
        self.photometric_episode_start_step = None
        self.photometric_recovery_streak = 0

    def _abort_photometric_quarantine(self, step, reason):
        pool = list(self.novel_quarantine or [])
        self.novel_quarantine = None
        self.novel_quarantine_route = None
        self.photometric_trigger_expert_id = None
        self.photometric_trigger_lab_center = None
        self.last_novel_quarantine_event = {
            "decision": reason,
            "start_step": (
                int(pool[0]["step"]) if pool else None
            ),
            "end_step": int(step),
            "size": len(pool),
            "route": "photometric_drift",
            "image_ids": [int(member["image_id"]) for member in pool],
        }

    def _photometric_trigger_center(self, trigger_expert_id):
        expert = next(
            (
                item
                for item in self.active_experts
                if (
                    item.get("expert_id") is not None
                    and int(item["expert_id"]) == int(trigger_expert_id)
                )
            ),
            None,
        )
        if expert is None or expert.get("lab_state") is None:
            return None
        return np.asarray(
            expert["lab_state"]["center"],
            dtype=np.float64,
        ).copy()

    def _photometric_pool_profile(
        self,
        pool,
        trigger_expert_id,
        trigger_lab_center,
    ):
        if trigger_lab_center is None:
            return {
                "consistent": False,
                "reason": "trigger_expert_missing",
            }
        features = [item.get("lab_feature") for item in pool]
        if any(feature is None for feature in features):
            return {
                "consistent": False,
                "reason": "lab_feature_missing",
            }

        if self.photometric_candidate_window_mode != "history":
            normalized = np.stack(
                [
                    np.asarray(feature, dtype=np.float64)
                    / LAB_FEATURE_SCALES
                    for feature in features
                ],
                axis=0,
            )
            pairwise_distances = np.linalg.norm(
                normalized[:, None, :] - normalized[None, :, :],
                axis=2,
            )
            medoid_index = int(pairwise_distances.mean(axis=1).argmin())
            local_distances = pairwise_distances[medoid_index]
            required = int(
                math.ceil(len(pool) * self.photometric_consistency_fraction)
            )
            lab_inlier_count = int(
                np.sum(
                    local_distances
                    <= self.photometric_local_lab_distance_threshold
                )
            )

            embeddings = torch.stack(
                [member["embedding"] for member in pool],
                dim=0,
            )
            similarities = embeddings @ embeddings.T
            dino_medoid_index = int(similarities.mean(dim=1).argmax())
            dino_medoid_similarities = similarities[dino_medoid_index]
            dino_inlier_count = int(
                torch.sum(
                    dino_medoid_similarities
                    >= self.photometric_dino_similarity_floor
                ).item()
            )

            local_center = np.median(normalized, axis=0)
            old_center = (
                np.asarray(trigger_lab_center, dtype=np.float64)
                / LAB_FEATURE_SCALES
            )
            separation = local_center - old_center
            l_separation = float(np.linalg.norm(separation[:2]))
            ab_separation = float(np.linalg.norm(separation[2:]))
            separated = bool(
                l_separation >= self.photometric_l_ewma_floor
                or ab_separation >= self.photometric_ab_ewma_floor
            )
            consistent = bool(
                separated
                and lab_inlier_count >= required
                and dino_inlier_count >= required
            )
            return {
                "consistent": consistent,
                "reason": (
                    "consistent"
                    if consistent
                    else "insufficient_fixed_scale_support"
                ),
                "window_mode": self.photometric_candidate_window_mode,
                "trigger_expert_id": int(trigger_expert_id),
                "evaluated_member_count": len(pool),
                "required_count": required,
                "lab_inlier_count": lab_inlier_count,
                "dino_inlier_count": dino_inlier_count,
                "local_lab_distance_threshold": (
                    self.photometric_local_lab_distance_threshold
                ),
                "local_distance_max": float(local_distances.max()),
                "local_distance_median": float(
                    np.median(local_distances)
                ),
                "l_separation": l_separation,
                "ab_separation": ab_separation,
                "selected_lab_medoid_image_id": int(
                    pool[medoid_index]["image_id"]
                ),
                "selected_dino_medoid_image_id": int(
                    pool[dino_medoid_index]["image_id"]
                ),
            }

        if self.photometric_relevant_history_enabled:
            normalized = np.stack(
                [
                    np.asarray(feature, dtype=np.float64)
                    / LAB_FEATURE_SCALES
                    for feature in features
                ],
                axis=0,
            )
            local_center = np.median(normalized, axis=0)
            local_mad = np.median(
                np.abs(normalized - local_center),
                axis=0,
            )
            local_scale = np.maximum(
                1.4826 * local_mad,
                self.drift_lab_noise_floor,
            )
            local_distances = np.linalg.norm(
                (normalized - local_center) / local_scale,
                axis=1,
            ) / math.sqrt(normalized.shape[1])
            required = int(
                math.ceil(len(pool) * self.photometric_consistency_fraction)
            )
            lab_inlier_count = int(
                np.sum(
                    local_distances
                    <= self.photometric_history_distance_threshold
                )
            )

            embeddings = torch.stack(
                [member["embedding"] for member in pool],
                dim=0,
            )
            similarities = embeddings @ embeddings.T
            medoid_index = int(similarities.mean(dim=1).argmax())
            medoid_similarities = similarities[medoid_index]
            dino_inlier_count = int(
                torch.sum(
                    medoid_similarities
                    >= self.photometric_dino_similarity_floor
                ).item()
            )

            old_center = (
                np.asarray(trigger_lab_center, dtype=np.float64)
                / LAB_FEATURE_SCALES
            )
            separation = local_center - old_center
            l_separation = float(np.linalg.norm(separation[:2]))
            ab_separation = float(np.linalg.norm(separation[2:]))
            separated = bool(
                l_separation >= self.photometric_l_ewma_floor
                or ab_separation >= self.photometric_ab_ewma_floor
            )
            consistent = bool(
                separated
                and lab_inlier_count >= required
                and dino_inlier_count >= required
            )
            return {
                "consistent": consistent,
                "reason": (
                    "consistent"
                    if consistent
                    else "insufficient_local_distribution_support"
                ),
                "trigger_expert_id": int(trigger_expert_id),
                "evaluated_member_count": len(pool),
                "required_count": required,
                "lab_inlier_count": lab_inlier_count,
                "dino_inlier_count": dino_inlier_count,
                "dino_similarity_floor": (
                    self.photometric_dino_similarity_floor
                ),
                "local_distance_max": float(local_distances.max()),
                "local_distance_median": float(
                    np.median(local_distances)
                ),
                "l_separation": l_separation,
                "ab_separation": ab_separation,
                "selected_medoid_image_id": int(
                    pool[medoid_index]["image_id"]
                ),
            }

        center = np.asarray(trigger_lab_center, dtype=np.float64)
        deltas = np.stack(
            [
                (np.asarray(feature, dtype=np.float64) - center)
                / LAB_FEATURE_SCALES
                for feature in features
            ],
            axis=0,
        )
        l_distances = np.linalg.norm(deltas[:, :2], axis=1)
        ab_distances = np.linalg.norm(deltas[:, 2:], axis=1)
        median_delta = np.median(deltas, axis=0)
        median_l = float(np.linalg.norm(median_delta[:2]))
        median_ab = float(np.linalg.norm(median_delta[2:]))
        dominant_channel = "l" if (
            median_l / self.photometric_l_ewma_floor
            >= median_ab / self.photometric_ab_ewma_floor
        ) else "ab"
        dominant_distances = (
            l_distances if dominant_channel == "l" else ab_distances
        )
        dominant_floor = (
            self.photometric_l_ewma_floor
            if dominant_channel == "l"
            else self.photometric_ab_ewma_floor
        )
        required = int(
            math.ceil(len(pool) * self.photometric_consistency_fraction)
        )
        shifted_count = int(np.sum(dominant_distances >= dominant_floor))
        direction_count = int(np.sum((deltas @ median_delta) > 0.0))
        return {
            "consistent": bool(
                shifted_count >= required and direction_count >= required
            ),
            "reason": "consistent" if (
                shifted_count >= required and direction_count >= required
            ) else "insufficient_persistence",
            "trigger_expert_id": int(trigger_expert_id),
            "dominant_channel": dominant_channel,
            "required_count": required,
            "shifted_count": shifted_count,
            "direction_count": direction_count,
            "median_l_distance": median_l,
            "median_ab_distance": median_ab,
        }

    @staticmethod
    def _optional_median(values):
        finite = [
            float(value)
            for value in values
            if value is not None and math.isfinite(float(value))
        ]
        return float(np.median(finite)) if finite else None

    def advance_drift_candidate_audit(
        self,
        step,
        image_id,
        dino_patch_features,
        lab_feature,
        ms_short_score,
        lab_drift_profile,
        channel_support,
        mature_match_fraction,
        recent_provisional_match_fraction,
    ):
        self.last_drift_candidate_audit_event = None
        if not self.drift_candidate_audit_enabled:
            return None

        feature = self._lab_feature(lab_feature)
        if feature is None:
            if self.drift_candidate_pool is not None:
                self.drift_candidate_pool = None
                self.last_drift_candidate_audit_event = {
                    "decision": "aborted_missing_lab",
                    "step": int(step),
                }
            return self.last_drift_candidate_audit_event

        member = {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": self._image_embedding(dino_patch_features),
            "lab_feature": feature,
            "ms_short_score": (
                float(ms_short_score) if ms_short_score is not None else None
            ),
            "channel_support": (
                float(channel_support) if channel_support is not None else None
            ),
            "mature_match_fraction": (
                float(mature_match_fraction)
                if mature_match_fraction is not None
                else None
            ),
            "recent_provisional_match_fraction": (
                float(recent_provisional_match_fraction)
                if recent_provisional_match_fraction is not None
                else None
            ),
        }

        if self.drift_candidate_pool is None:
            if not (
                lab_drift_profile is not None
                and lab_drift_profile.get("drift_confirmed", False)
            ):
                return None
            self.drift_candidate_pool = {
                "start_step": int(step),
                "trigger_expert_id": int(lab_drift_profile["expert_id"]),
                "members": [member],
            }
            self.last_drift_candidate_audit_event = {
                "decision": "opened",
                "start_step": int(step),
                "trigger_expert_id": int(lab_drift_profile["expert_id"]),
                "pool_size": 1,
            }
            return self.last_drift_candidate_audit_event

        pool = self.drift_candidate_pool
        pool["members"].append(member)
        if len(pool["members"]) < self.drift_candidate_pool_size:
            self.last_drift_candidate_audit_event = {
                "decision": "collecting",
                "start_step": int(pool["start_step"]),
                "trigger_expert_id": int(pool["trigger_expert_id"]),
                "pool_size": len(pool["members"]),
            }
            return self.last_drift_candidate_audit_event

        self.drift_candidate_pool = None
        ranked = sorted(
            pool["members"],
            key=lambda item: (
                item["ms_short_score"] is None,
                item["ms_short_score"]
                if item["ms_short_score"] is not None
                else float("inf"),
                item["step"],
            ),
        )
        selected = ranked[:self.drift_candidate_keep]
        embeddings = torch.stack([item["embedding"] for item in selected])
        similarities = embeddings @ embeddings.T
        pair_indices = torch.triu_indices(
            len(selected),
            len(selected),
            offset=1,
        )
        pair_similarities = similarities[
            pair_indices[0],
            pair_indices[1],
        ]
        candidate_embedding = F.normalize(
            embeddings.mean(dim=0),
            dim=0,
        )

        lab_features = np.stack(
            [item["lab_feature"] for item in selected],
            axis=0,
        )
        candidate_center = np.median(lab_features, axis=0)
        within_distances = [
            self._lab_distances(feature, candidate_center)["total_distance"]
            for feature in lab_features
        ]

        trigger_expert = next(
            (
                expert
                for expert in self.active_experts
                if int(expert["expert_id"])
                == int(pool["trigger_expert_id"])
            ),
            None,
        )
        base_event = {
            "decision": "audited",
            "start_step": int(pool["start_step"]),
            "end_step": int(step),
            "trigger_expert_id": int(pool["trigger_expert_id"]),
            "pool_image_ids": [
                int(item["image_id"]) for item in pool["members"]
            ],
            "selected_image_ids": [
                int(item["image_id"]) for item in selected
            ],
            "selected_ms_short_scores": [
                item["ms_short_score"] for item in selected
            ],
            "dino_consistency_min": float(pair_similarities.min()),
            "dino_consistency_mean": float(pair_similarities.mean()),
            "lab_candidate_center": candidate_center.tolist(),
            "lab_within_distance_median": float(
                np.median(within_distances)
            ),
            "lab_within_distance_max": float(np.max(within_distances)),
            "channel_support_median": self._optional_median(
                [item["channel_support"] for item in selected]
            ),
            "mature_match_fraction_median": self._optional_median(
                [item["mature_match_fraction"] for item in selected]
            ),
            "recent_provisional_match_fraction_median": (
                self._optional_median(
                    [
                        item["recent_provisional_match_fraction"]
                        for item in selected
                    ]
                )
            ),
        }
        if trigger_expert is None or trigger_expert.get("lab_state") is None:
            base_event["decision"] = "audited_expert_missing"
            self.last_drift_candidate_audit_event = base_event
            return base_event

        expert_lab_state = trigger_expert["lab_state"]
        center_distances = self._lab_distances(
            candidate_center,
            np.asarray(expert_lab_state["center"], dtype=np.float64),
        )
        old_l_scale = 1.4826 * float(
            expert_lab_state["l_baseline_mad"]
        )
        old_ab_scale = 1.4826 * float(
            expert_lab_state["ab_baseline_mad"]
        )
        old_total_scale = math.hypot(old_l_scale, old_ab_scale)
        separation_denominator = max(
            old_total_scale,
            float(np.median(within_distances)),
            self.drift_lab_noise_floor,
        )

        cluster = self.clusters[int(trigger_expert["cluster_id"])]
        expert_embedding = F.normalize(cluster["embedding_sum"], dim=0)
        base_event.update(
            {
                "lab_center_l_distance": center_distances["l_distance"],
                "lab_center_ab_distance": center_distances["ab_distance"],
                "lab_center_total_distance": center_distances[
                    "total_distance"
                ],
                "lab_old_total_scale": float(old_total_scale),
                "lab_separation_denominator": float(
                    separation_denominator
                ),
                "lab_separation_ratio": float(
                    center_distances["total_distance"]
                    / separation_denominator
                ),
                "dino_to_trigger_expert_similarity": float(
                    candidate_embedding @ expert_embedding
                ),
            }
        )
        self.last_drift_candidate_audit_event = base_event
        return base_event

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
        lifecycle_records = []

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
            lifecycle_records.append(
                {
                    "expert": expert,
                    "expert_id": expert_id,
                    "support": support,
                    "support_gap": support_gap,
                    "ttl_before": ttl_before,
                    "patience_before": patience_before,
                    "last_supported_step_before": (
                        last_supported_step_before
                    ),
                    "signal_decision": signal_decision,
                    "tentative_delete": expert["ttl"] <= 0,
                }
            )

        tentative_deletions = [
            record
            for record in lifecycle_records
            if record["tentative_delete"]
        ]
        natural_survivor_count = (
            len(lifecycle_records) - len(tentative_deletions)
        )
        protect_count = max(
            0,
            self.minimum_active_experts - natural_survivor_count,
        )
        protected_ids = {
            int(record["expert_id"])
            for record in sorted(
                tentative_deletions,
                key=lambda record: (
                    record["support"] is not None,
                    (
                        float(record["support"])
                        if record["support"] is not None
                        else -1.0
                    ),
                    int(record["expert"]["last_supported_step"]),
                    int(record["expert"]["support_count"]),
                    int(record["expert"]["admission_step"]),
                ),
                reverse=True,
            )[:protect_count]
        }

        for record in lifecycle_records:
            expert = record["expert"]
            expert_id = record["expert_id"]
            deleted_this_step = False
            coverage_floor_retained = bool(
                record["tentative_delete"]
                and int(expert_id) in protected_ids
            )
            if coverage_floor_retained:
                expert["ttl"] = 1
                expert["deletion_step"] = None
                expert["last_event"] = "coverage_floor_retained"
                survivors.append(expert)
            elif record["tentative_delete"]:
                expert["ttl"] = 0
                expert["deletion_step"] = step
                expert["last_event"] = "deleted"
                deleted_this_step = True

                cluster = self.clusters[expert["cluster_id"]]
                if cluster["active_expert_id"] == expert_id:
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
                    "channel_support": record["support"],
                    "support_threshold": support_threshold,
                    "support_available": record["support"] is not None,
                    "threshold_available": support_threshold is not None,
                    "support_gap": record["support_gap"],
                    "ttl_before": record["ttl_before"],
                    "ttl_after": int(expert["ttl"]),
                    "patience_before": record["patience_before"],
                    "patience_after": int(expert["patience"]),
                    "last_supported_step_before": (
                        record["last_supported_step_before"]
                    ),
                    "last_supported_step_after": int(
                        expert["last_supported_step"]
                    ),
                    "signal_decision": record["signal_decision"],
                    "decision": expert["last_event"],
                    "refreshed": (
                        record["signal_decision"] == "ttl_refreshed"
                    ),
                    "deleted_this_step": deleted_this_step,
                    "alive_after": not deleted_this_step,
                    "coverage_floor_retained": coverage_floor_retained,
                    "minimum_active_experts": self.minimum_active_experts,
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

    def _quarantine_member(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        ms_short_score,
        ms_short_ratio,
        channel_support,
        provisional_channel_support,
        memory_patch_ids=None,
        lab_feature=None,
        candidate_route="new_distribution",
        historical_ms_threshold=None,
    ):
        return {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": self._image_embedding(dino_patch_features),
            "quality": float(provisional_channel_support or 0.0),
            "ms_score": ms_score,
            "ms_short_score": ms_short_score,
            "ms_short_ratio": ms_short_ratio,
            "channel_support": channel_support,
            "provisional_channel_support": provisional_channel_support,
            "candidate_route": candidate_route,
            "historical_ms_threshold": historical_ms_threshold,
            "historical_low_risk": bool(
                ms_score is not None
                and historical_ms_threshold is not None
                and ms_score <= historical_ms_threshold
            ),
            "memory_patch_ids": list(memory_patch_ids or []),
            "lab_feature": self._lab_feature(lab_feature),
        }

    def _select_irbank_members(self, pool):
        eligible = [
            member for member in pool if member.get("memory_patch_ids")
        ]
        if not eligible:
            return []
        keep = max(
            1,
            int(math.ceil(len(eligible) * self.novel_memory_keep_fraction)),
        )
        return sorted(
            eligible,
            key=lambda item: item["ms_short_score"],
        )[:keep]

    def _replace_budgeted_expert(self, expert_id, step):
        expert = next(
            (
                item
                for item in self.active_experts
                if int(item["expert_id"]) == int(expert_id)
            ),
            None,
        )
        if expert is None:
            return False
        self.active_experts = [
            item
            for item in self.active_experts
            if int(item["expert_id"]) != int(expert_id)
        ]
        expert["ttl"] = 0
        expert["deletion_step"] = int(step)
        expert["last_event"] = "replaced_photometric_budget"
        cluster = self.clusters[expert["cluster_id"]]
        if cluster["active_expert_id"] == expert["expert_id"]:
            cluster["active_expert_id"] = None
        self.retired_experts.append(expert)
        self.last_budget_replacement = {
            "expert_id": int(expert_id),
            "step": int(step),
            "reason": "photometric_coverage_replacement",
        }
        return True

    def _admit_quarantined_member(
        self,
        member,
        step,
        cluster_members=None,
        admission_route="new_distribution",
        replacement_expert_id=None,
    ):
        if len(self.active_experts) >= self.committee_cap:
            can_replace = bool(
                (
                    self.photometric_candidate_enabled
                    or self.photometric_relevant_history_enabled
                )
                and self.photometric_budget_replacement
                and admission_route.startswith("photometric_drift")
                and replacement_expert_id is not None
            )
            if not can_replace or not self._replace_budgeted_expert(
                replacement_expert_id,
                step,
            ):
                return None, None

        cluster_members = list(cluster_members or [member])
        cluster_id = len(self.clusters)
        self.clusters.append(
            {
                "embedding_sum": torch.stack(
                    [item["embedding"] for item in cluster_members]
                ).sum(dim=0),
                "members": cluster_members,
                "active_expert_id": self.next_expert_id,
            }
        )
        expert = {
            "expert_id": self.next_expert_id,
            "image_id": member["image_id"],
            "cluster_id": cluster_id,
            "admission_step": int(step),
            "cluster_support_at_admission": len(cluster_members),
            "max_support_gap": 0,
            "last_supported_step": int(step),
            "support_count": len(cluster_members),
            "patience": self.base_ttl,
            "ttl": self.base_ttl,
            "deletion_step": None,
            "last_event": "admitted",
            "last_channel_support": None,
            "representative_mode": self.representative_mode,
            "admission_ttl_mode": self.admission_ttl_mode,
            "admission_route": admission_route,
            "lab_state": (
                self._build_lab_state(cluster_members)
                if admission_route.startswith("photometric_drift")
                else None
            ),
            "admission_candidate_ms_score": member["ms_score"],
            "admission_candidate_channel_support": member[
                "channel_support"
            ],
        }
        self.active_experts.append(expert)
        self.admitted_count += 1
        self.next_expert_id += 1
        return cluster_id, expert["expert_id"]

# 更新候选池
    def _advance_novel_quarantine(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        ms_short_score,
        ms_short_ratio,
        channel_support,
        provisional_channel_support,
        trigger_route,
        memory_patch_ids=None,
        lab_feature=None,
        photometric_seed_members=None,
        photometric_trigger_expert_id=None,
        photometric_trigger_lab_center=None,
        historical_ms_threshold=None,
    ):
        self.last_novel_quarantine_event = None
        if self.novel_admission_mode not in {
            "quarantine",
            "provisional",
            "soft_confirmed",
            "channel_irbank",
            "channel_gate",
        }:
            return None, None
        if ms_short_score is None or ms_short_ratio is None:
            return None, None
        member = self._quarantine_member(
            step,
            image_id,
            dino_patch_features,
            ms_score,
            ms_short_score,
            ms_short_ratio,
            channel_support,
            provisional_channel_support,
            memory_patch_ids,
            lab_feature,
            candidate_route=(
                trigger_route
                or self.novel_quarantine_route
                or "new_distribution"
            ),
            historical_ms_threshold=historical_ms_threshold,
        )
        fixed_scale_photometric = bool(
            (
                trigger_route == "photometric_drift"
                or self.novel_quarantine_route == "photometric_drift"
            )
            and self.photometric_candidate_window_mode != "history"
        )
        if fixed_scale_photometric and not member["historical_low_risk"]:
            if self.novel_quarantine is not None:
                self.last_novel_quarantine_event = {
                    "decision": "waiting_low_risk",
                    "start_step": int(self.novel_quarantine[0]["step"]),
                    "size": len(self.novel_quarantine),
                    "route": self.novel_quarantine_route,
                    "skipped_image_id": int(image_id),
                }
            return None, None
        if self.novel_quarantine is None and trigger_route is None:
            if self.confirmed_provisional is not None:
                if self.novel_admission_mode == "channel_irbank":
                    candidates = self.confirmed_provisional["candidates"]
                    candidates.append(member)
                    del candidates[:-self.novel_memory_window]
                    selected = self._select_irbank_members(candidates)
                    self.confirmed_provisional["members"] = selected
                    if any(
                        selected_member["image_id"] == image_id
                        for selected_member in selected
                    ):
                        self.confirmed_provisional["ttl"] = (
                            self.novel_confirmed_ttl
                        )
                    else:
                        self.confirmed_provisional["ttl"] -= 1
                else:
                    self.confirmed_provisional["ttl"] -= 1
                if self.confirmed_provisional["ttl"] <= 0:
                    self.confirmed_provisional = None
            return None, None
        if self.novel_quarantine is None:
            if trigger_route is None:
                return None, None
            self.confirmed_provisional = None
            self.novel_quarantine_route = trigger_route
            self.photometric_trigger_expert_id = (
                int(photometric_trigger_expert_id)
                if trigger_route == "photometric_drift"
                else None
            )
            self.photometric_trigger_lab_center = (
                np.asarray(
                    photometric_trigger_lab_center,
                    dtype=np.float64,
                ).copy()
                if photometric_trigger_lab_center is not None
                else self._photometric_trigger_center(
                    self.photometric_trigger_expert_id
                )
                if self.photometric_trigger_expert_id is not None
                else None
            )
            self.novel_quarantine = (
                [
                    {
                        **seed_member,
                        "candidate_route": "photometric_drift",
                    }
                    for seed_member in (photometric_seed_members or [])
                ]
                + [member]
                if trigger_route == "photometric_drift"
                else [member]
            )
            unique_members = {}
            for pool_member in self.novel_quarantine:
                unique_members[int(pool_member["image_id"])] = pool_member
            self.novel_quarantine = list(unique_members.values())
            self.last_novel_quarantine_event = {
                "decision": "opened",
                "start_step": int(self.novel_quarantine[0]["step"]),
                "size": len(self.novel_quarantine),
                "route": self.novel_quarantine_route,
                "image_ids": [
                    int(item["image_id"])
                    for item in self.novel_quarantine
                ],
            }
            return None, None

        if self.novel_quarantine_route == "photometric_drift":
            trigger_route = None

        if (
            trigger_route is not None
            and ms_short_ratio
            < self.novel_quarantine[0]["ms_short_ratio"]
        ):
            self.novel_quarantine = (
                [
                    {
                        **seed_member,
                        "candidate_route": "photometric_drift",
                    }
                    for seed_member in (photometric_seed_members or [])
                ]
                + [member]
                if trigger_route == "photometric_drift"
                else [member]
            )
            unique_members = {}
            for pool_member in self.novel_quarantine:
                unique_members[int(pool_member["image_id"])] = pool_member
            self.novel_quarantine = list(unique_members.values())
            self.novel_quarantine_route = trigger_route
            self.photometric_trigger_expert_id = (
                int(photometric_trigger_expert_id)
                if trigger_route == "photometric_drift"
                else None
            )
            self.photometric_trigger_lab_center = (
                np.asarray(
                    photometric_trigger_lab_center,
                    dtype=np.float64,
                ).copy()
                if photometric_trigger_lab_center is not None
                else self._photometric_trigger_center(
                    self.photometric_trigger_expert_id
                )
                if self.photometric_trigger_expert_id is not None
                else None
            )
            self.last_novel_quarantine_event = {
                "decision": "restarted",
                "start_step": int(self.novel_quarantine[0]["step"]),
                "size": len(self.novel_quarantine),
                "route": self.novel_quarantine_route,
                "image_ids": [
                    int(item["image_id"])
                    for item in self.novel_quarantine
                ],
            }
            return None, None

        self.novel_quarantine.append(member)
        if self.novel_quarantine_route == "photometric_drift":
            if self.photometric_candidate_window_mode == "fixed_low_risk":
                required_steps = self.photometric_fixed_low_risk_members
            elif (
                self.photometric_candidate_window_mode
                == "dynamic_low_risk"
            ):
                required_steps = (
                    self.photometric_dynamic_min_low_risk_members
                )
            elif self.photometric_relevant_history_enabled:
                required_steps = max(
                    self.photometric_min_low_risk_members,
                    2,
                )
            else:
                required_steps = self.photometric_quarantine_steps
        else:
            required_steps = self.novel_quarantine_steps
        if len(self.novel_quarantine) < required_steps:
            self.last_novel_quarantine_event = {
                "decision": "collecting",
                "start_step": self.novel_quarantine[0]["step"],
                "size": len(self.novel_quarantine),
                "required_steps": required_steps,
                "route": self.novel_quarantine_route,
            }
            return None, None

        if (
            self.novel_quarantine_route == "photometric_drift"
            and self.photometric_candidate_window_mode
            == "dynamic_low_risk"
        ):
            provisional_profile = self._photometric_pool_profile(
                self.novel_quarantine,
                self.photometric_trigger_expert_id,
                self.photometric_trigger_lab_center,
            )
            if (
                not provisional_profile["consistent"]
                and len(self.novel_quarantine)
                < self.photometric_dynamic_max_low_risk_members
            ):
                self.last_novel_quarantine_event = {
                    "decision": "collecting_inconsistent",
                    "start_step": int(self.novel_quarantine[0]["step"]),
                    "size": len(self.novel_quarantine),
                    "required_steps": (
                        self.photometric_dynamic_max_low_risk_members
                    ),
                    "route": self.novel_quarantine_route,
                    "photometric_profile": provisional_profile,
                }
                return None, None

        pool = self.novel_quarantine
        pool_route = self.novel_quarantine_route
        pool_trigger_expert_id = self.photometric_trigger_expert_id
        pool_trigger_lab_center = self.photometric_trigger_lab_center
        self.novel_quarantine = None
        self.novel_quarantine_route = None
        self.photometric_trigger_expert_id = None
        self.photometric_trigger_lab_center = None
        median_ratio = float(
            np.median([item["ms_short_ratio"] for item in pool])
        )
        if (
            pool_route != "photometric_drift"
            and median_ratio > self.novel_ratio_threshold
        ):
            self.last_novel_quarantine_event = {
                "decision": "rejected",
                "start_step": pool[0]["step"],
                "end_step": int(step),
                "size": len(pool),
                "median_ratio": median_ratio,
                "selected_image_id": None,
                "route": pool_route,
            }
            return None, None

        photometric_profile = None
        photometric_low_risk_members = None
        if pool_route == "photometric_drift":
            photometric_low_risk_members = [
                member
                for member in pool
                if member.get("historical_low_risk", False)
            ]
            if (
                len(photometric_low_risk_members)
                < self.photometric_min_low_risk_members
            ):
                self.photometric_support_streak = 0
                self.last_novel_quarantine_event = {
                    "decision": (
                        "rejected_photometric_insufficient_low_risk"
                    ),
                    "start_step": pool[0]["step"],
                    "end_step": int(step),
                    "size": len(pool),
                    "median_ratio": median_ratio,
                    "selected_image_id": None,
                    "route": pool_route,
                    "photometric_profile": {
                        "consistent": False,
                        "reason": "insufficient_low_risk",
                        "trigger_expert_id": pool_trigger_expert_id,
                        "evaluated_member_count": len(
                            photometric_low_risk_members
                        ),
                    },
                    "low_risk_count": len(photometric_low_risk_members),
                    "required_low_risk_count": (
                        self.photometric_min_low_risk_members
                    ),
                    "low_risk_image_ids": [
                        int(member["image_id"])
                        for member in photometric_low_risk_members
                    ],
                }
                return None, None
            photometric_profile = self._photometric_pool_profile(
                photometric_low_risk_members,
                pool_trigger_expert_id,
                pool_trigger_lab_center,
            )
            if not photometric_profile["consistent"]:
                self.photometric_support_streak = 0
                self.last_novel_quarantine_event = {
                    "decision": "rejected_photometric_inconsistent",
                    "start_step": pool[0]["step"],
                    "end_step": int(step),
                    "size": len(pool),
                    "median_ratio": median_ratio,
                    "selected_image_id": None,
                    "route": pool_route,
                    "photometric_profile": photometric_profile,
                    "low_risk_count": len(
                        photometric_low_risk_members
                    ),
                    "required_low_risk_count": (
                        self.photometric_min_low_risk_members
                    ),
                    "low_risk_image_ids": [
                        int(member["image_id"])
                        for member in photometric_low_risk_members
                    ],
                }
                return None, None

        admission_pool = (
            photometric_low_risk_members
            if photometric_low_risk_members is not None
            else pool
        )
        representative_candidates = (
            admission_pool
            if (
                pool_route == "photometric_drift"
                and self.photometric_candidate_window_mode != "history"
            )
            else admission_pool[-self.novel_quarantine_tail:]
        )
        representative = min(
            representative_candidates,
            key=lambda item: item["ms_short_score"],
        )
        if self.novel_admission_mode in {
            "soft_confirmed",
            "channel_irbank",
            "channel_gate",
        }:
            if self.novel_admission_mode in {
                "channel_irbank",
                "channel_gate",
            }:
                representatives = self._select_irbank_members(
                    photometric_low_risk_members
                    if pool_route == "photometric_drift"
                    else pool
                )
                if (
                    pool_route == "photometric_drift"
                    and self.photometric_candidate_window_mode != "history"
                    and len(representatives) < 2
                ):
                    representatives = []
            else:
                representatives = sorted(
                    admission_pool[-self.novel_quarantine_tail:],
                    key=lambda item: item["ms_short_score"],
                )[:self.novel_confirmed_members]
            if not representatives:
                if pool_route == "photometric_drift":
                    self.photometric_support_streak = 0
                self.last_novel_quarantine_event = {
                    "decision": "rejected_no_reliable_patches",
                    "start_step": pool[0]["step"],
                    "end_step": int(step),
                    "size": len(pool),
                    "median_ratio": median_ratio,
                    "selected_image_id": None,
                    "route": pool_route,
                    "photometric_profile": photometric_profile,
                    "low_risk_count": (
                        len(photometric_low_risk_members)
                        if photometric_low_risk_members is not None
                        else None
                    ),
                    "required_low_risk_count": (
                        self.photometric_min_low_risk_members
                        if pool_route == "photometric_drift"
                        else None
                    ),
                    "low_risk_image_ids": (
                        [
                            int(member["image_id"])
                            for member in photometric_low_risk_members
                        ]
                        if photometric_low_risk_members is not None
                        else []
                    ),
                }
                return None, None
            if self.novel_admission_mode == "channel_gate":
                representative = representatives[0]
                cluster_id, expert_id = self._admit_quarantined_member(
                    representative,
                    step,
                    cluster_members=(
                        representatives
                        if pool_route == "photometric_drift"
                        else [representative]
                    ),
                    admission_route=f"{pool_route}_channel_gate",
                    replacement_expert_id=(
                        pool_trigger_expert_id
                        if pool_route == "photometric_drift"
                        else None
                    ),
                )
                self.last_novel_quarantine_event = {
                    "decision": (
                        "confirmed_channel_gate"
                        if expert_id is not None
                        else "cap_rejected"
                    ),
                    "start_step": pool[0]["step"],
                    "end_step": int(step),
                    "size": len(pool),
                    "median_ratio": median_ratio,
                    "selected_image_id": representative["image_id"],
                    "selected_ms_short": representative["ms_short_score"],
                    "confirmed_image_ids": [
                        int(member["image_id"])
                        for member in representatives
                    ],
                    "confirmed_patch_count": sum(
                        len(member.get("memory_patch_ids", []))
                        for member in representatives
                    ),
                    "route": pool_route,
                    "photometric_profile": photometric_profile,
                    "low_risk_count": (
                        len(photometric_low_risk_members)
                        if photometric_low_risk_members is not None
                        else None
                    ),
                    "required_low_risk_count": (
                        self.photometric_min_low_risk_members
                        if pool_route == "photometric_drift"
                        else None
                    ),
                    "low_risk_image_ids": (
                        [
                            int(member["image_id"])
                            for member in photometric_low_risk_members
                        ]
                        if photometric_low_risk_members is not None
                        else []
                    ),
                }
                if pool_route == "photometric_drift":
                    self.photometric_armed_expert_id = None
                    self.photometric_support_streak = 0
                return cluster_id, expert_id
            self.confirmed_provisional = {
                "members": representatives,
                "candidates": list(pool),
                "start_step": pool[0]["step"],
                "confirmation_step": int(step),
                "ttl": self.novel_confirmed_ttl,
            }
            self.last_novel_quarantine_event = {
                "decision": (
                    "confirmed_irbank"
                    if self.novel_admission_mode == "channel_irbank"
                    else "confirmed_soft"
                ),
                "start_step": pool[0]["step"],
                "end_step": int(step),
                "size": len(pool),
                "median_ratio": median_ratio,
                "selected_image_id": representative["image_id"],
                "selected_ms_short": representative["ms_short_score"],
                "confirmed_weight": self.novel_confirmed_weight,
                "confirmed_ttl": self.novel_confirmed_ttl,
                "confirmed_image_ids": [
                    int(member["image_id"])
                    for member in representatives
                ],
                "confirmed_patch_count": sum(
                    len(member.get("memory_patch_ids", []))
                    for member in representatives
                ),
                "route": pool_route,
                "photometric_profile": photometric_profile,
                "low_risk_count": (
                    len(photometric_low_risk_members)
                    if photometric_low_risk_members is not None
                    else None
                ),
                "required_low_risk_count": (
                    self.photometric_min_low_risk_members
                    if pool_route == "photometric_drift"
                    else None
                ),
                "low_risk_image_ids": (
                    [
                        int(member["image_id"])
                        for member in photometric_low_risk_members
                    ]
                    if photometric_low_risk_members is not None
                    else []
                ),
            }
            return None, None
        cluster_id, expert_id = self._admit_quarantined_member(
            representative,
            step,
            cluster_members=(
                admission_pool
                if pool_route == "photometric_drift"
                else [representative]
            ),
            admission_route=(
                f"{pool_route}_provisional"
                if self.novel_admission_mode == "provisional"
                else f"{pool_route}_quarantine"
            ),
            replacement_expert_id=(
                pool_trigger_expert_id
                if pool_route == "photometric_drift"
                else None
            ),
        )
        self.last_novel_quarantine_event = {
            "decision": "admitted" if expert_id is not None else "cap_rejected",
            "start_step": pool[0]["step"],
            "end_step": int(step),
            "size": len(pool),
            "median_ratio": median_ratio,
            "selected_image_id": representative["image_id"],
            "selected_ms_short": representative["ms_short_score"],
            "route": pool_route,
            "photometric_profile": photometric_profile,
            "low_risk_count": (
                len(photometric_low_risk_members)
                if photometric_low_risk_members is not None
                else None
            ),
            "required_low_risk_count": (
                self.photometric_min_low_risk_members
                if pool_route == "photometric_drift"
                else None
            ),
            "low_risk_image_ids": (
                [
                    int(member["image_id"])
                    for member in photometric_low_risk_members
                ]
                if photometric_low_risk_members is not None
                else []
            ),
        }
        if pool_route == "photometric_drift":
            self.photometric_armed_expert_id = None
            self.photometric_support_streak = 0
        return cluster_id, expert_id


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
        ms_short_score,
        ms_short_ratio,
        memory_patch_ids,
        lab_feature,
        lab_drift_profile,
        current_expert_supports=None,
    ):
        current_expert_supports = current_expert_supports or {}
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
        (
            photometric_candidate,
            photometric_trigger_expert_id,
        ) = self._advance_photometric_gate(
            lab_drift_profile,
            support_threshold,
            current_expert_supports,
            step=step,
        )
        photometric_episode_pending = bool(
            self.photometric_episode_enabled
            and self.photometric_episode_active
            and self.photometric_episode_pending
        )
        if photometric_episode_pending:
            photometric_trigger_expert_id = (
                self.photometric_episode_reference_expert_id
            )
        photometric_trigger_lab_center = (
            self.photometric_episode_lab_center
            if photometric_episode_pending
            else lab_drift_profile.get("lab_center")
            if photometric_candidate and lab_drift_profile is not None
            else None
        )
        photometric_pool_active = bool(
            self.novel_quarantine is not None
            and self.novel_quarantine_route == "photometric_drift"
        )
        candidate_route = (
            "photometric_drift"
            if (
                photometric_pool_active
                or photometric_candidate
                or photometric_episode_pending
            )
            else "new_distribution"
            if novel_candidate
            else "legacy"
            if legacy_candidate
            else None
        )

        current_member = self._quarantine_member(
            step,
            image_id,
            dino_patch_features,
            ms_score,
            ms_short_score,
            ms_short_ratio,
            channel_support,
            provisional_channel_support,
            memory_patch_ids,
            lab_feature,
            candidate_route=(candidate_route or "history"),
            historical_ms_threshold=ms_threshold,
        )
        if self.photometric_pool_abort_reason is not None:
            self._abort_photometric_quarantine(
                step,
                self.photometric_pool_abort_reason,
            )
            self.photometric_prebuffer.append(current_member)
            prebuffer_size = (
                self.photometric_history_max
                if self.photometric_relevant_history_enabled
                else self.photometric_prebuffer_size
            )
            del self.photometric_prebuffer[:-prebuffer_size]
            return True, None, None, "photometric_drift"
        photometric_seed_members = []
        if (
            candidate_route == "photometric_drift"
            and not photometric_pool_active
            and self.photometric_candidate_window_mode == "history"
        ):
            if self.photometric_relevant_history_enabled:
                relevant_ids = set(
                    int(image_id)
                    for image_id in lab_drift_profile.get(
                        "relevant_image_ids", []
                    )
                )
                photometric_seed_members = [
                    member
                    for member in self.photometric_prebuffer
                    if int(member["image_id"]) in relevant_ids
                ]
            else:
                history_count = max(self.photometric_prebuffer_size - 1, 0)
                photometric_seed_members = list(
                    self.photometric_prebuffer[-history_count:]
                    if history_count
                    else []
                )

        quarantine_cluster_id, quarantine_expert_id = (
            self._advance_novel_quarantine(
                step,
                image_id,
                dino_patch_features,
                ms_score,
                ms_short_score,
                ms_short_ratio,
                channel_support,
                provisional_channel_support,
                (
                    candidate_route
                    if (
                        candidate_route
                        in {"new_distribution", "photometric_drift"}
                        and not photometric_pool_active
                    )
                    else None
                ),
                memory_patch_ids,
                lab_feature,
                photometric_seed_members,
                photometric_trigger_expert_id,
                photometric_trigger_lab_center,
                historical_ms_threshold=ms_threshold,
            )
        )
        if (
            self.photometric_episode_enabled
            and self.novel_quarantine is not None
            and self.novel_quarantine_route == "photometric_drift"
        ):
            self.photometric_episode_pending = False
        self.photometric_prebuffer.append(current_member)
        prebuffer_size = (
            self.photometric_history_max
            if self.photometric_relevant_history_enabled
            else self.photometric_prebuffer_size
        )
        del self.photometric_prebuffer[:-prebuffer_size]
        quarantine_route = (
            self.last_novel_quarantine_event.get("route")
            if self.last_novel_quarantine_event is not None
            else None
        )
        if quarantine_expert_id is not None:
            return (
                True,
                quarantine_cluster_id,
                quarantine_expert_id,
                quarantine_route,
            )
        if (
            self.last_novel_quarantine_event is not None
            and self.last_novel_quarantine_event.get("decision")
            in {"opened", "collecting", "restarted"}
        ):
            return True, None, None, quarantine_route
        if (
            candidate_route in {"new_distribution", "photometric_drift"}
            and self.novel_admission_mode
            in {
                "quarantine",
                "provisional",
                "soft_confirmed",
                "channel_irbank",
                "channel_gate",
            }
        ):
            return True, None, None, candidate_route

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
        ms_short_score=None,
        ms_short_ratio=None,
        memory_patch_ids=None,
        lab_feature=None,
        lab_drift_profile=None,
        current_expert_supports=None,
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
        ms_short_score = self._optional_score(
            "ms_short_score",
            ms_short_score,
        )
        ms_short_ratio = self._optional_score(
            "ms_short_ratio",
            ms_short_ratio,
        )
        if memory_patch_ids is not None:
            patch_count = int(dino_patch_features.shape[0])
            memory_patch_ids = sorted(
                {int(patch_id) for patch_id in memory_patch_ids}
            )
            if any(
                patch_id < 0 or patch_id >= patch_count
                for patch_id in memory_patch_ids
            ):
                raise ValueError("memory_patch_ids contains an invalid patch")

# 计算历史限制
        (
            ms_threshold,
            support_threshold,
        ) = self._historical_thresholds()
        (
            novel_ms_threshold,
            novel_support_threshold,
        ) = self._novel_thresholds()
        self.last_budget_replacement = None

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
        self.last_bootstrap_event = None
        if (
            self.cold_start_mode == "adaptive"
            and self.admitted_count == 0
            and not self.active_experts
        ):
            (
                is_candidate,
                cluster_id,
                admitted_expert_id,
                candidate_route,
            ) = self._advance_cold_start(
                step,
                int(image_id),
                dino_patch_features,
                ms_score,
                channel_support,
                lab_feature,
            )
        else:
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
                ms_short_score,
                ms_short_ratio,
                memory_patch_ids,
                lab_feature,
                lab_drift_profile,
                current_expert_supports=(current_expert_supports or {}),
            )

        if self.last_budget_replacement is not None:
            replaced_expert_id = int(
                self.last_budget_replacement["expert_id"]
            )
            if replaced_expert_id not in deleted_expert_ids:
                deleted_expert_ids.append(replaced_expert_id)
            expert_lifecycle_events.append(
                {
                    "step": int(step),
                    "expert_id": replaced_expert_id,
                    "decision": "replaced_photometric_budget",
                    "signal_decision": "replaced_photometric_budget",
                    "deleted_this_step": True,
                    "alive_after": False,
                }
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
                    "admission_candidate_image_id": int(
                        admitted_expert["image_id"]
                    ),
                    "admission_candidate_ms_score": admitted_expert.get(
                        "admission_candidate_ms_score", ms_score
                    ),
                    "admission_ms_threshold": ms_threshold,
                    "admission_candidate_channel_support": (
                        admitted_expert.get(
                            "admission_candidate_channel_support",
                            channel_support,
                        )
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
            "admission_route": (
                next(
                    expert["admission_route"]
                    for expert in self.active_experts
                    if expert["expert_id"] == admitted_expert_id
                )
                if admitted_expert_id is not None
                else None
            ),
            "novel_quarantine": self.last_novel_quarantine_event,
            "photometric_gate": self.last_photometric_gate,
            "budget_replacement": self.last_budget_replacement,
            "bootstrap": self.last_bootstrap_event,
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

