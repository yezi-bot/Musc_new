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
    cold_start_provisional_enabled=False,
    cold_start_provisional_confirm_steps=2,
    cold_start_provisional_max_wait_steps=4,
    cold_start_provisional_weight=0.25,
    cold_start_provisional_min_channel_support=0.05,
    cold_start_confidence_history=16,
    cold_start_reference_confidence_mode="state",
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
    photometric_fitness_history_enabled=False,
    photometric_history_min=3,
    photometric_history_max=16,
    photometric_history_distance_threshold=3.0,
    photometric_dino_similarity_floor=0.9,
    photometric_history_reentry_enabled=False,
    photometric_history_reentry_members=2,
    photometric_history_median_risk_max=1.25,
    photometric_history_extreme_risk_max=2.0,
    photometric_state_registry_cap=None,
    photometric_state_distribution_enabled=None,
    photometric_temporary_state_enabled=None,
    photometric_adaptive_candidate_evidence=False,
    photometric_stable_lab_anchor_enabled=False,
    photometric_denstream_support_enabled=False,
    photometric_denstream_decay=0.9,
    photometric_denstream_mature_support=0.5,
    photometric_denstream_discard_support=0.05,
    photometric_budget_replacement=True,
    photometric_budget_admission_mode="channel_only",
    legacy_budget_replacement=True,
    budget_coverage_window=32,
    budget_min_new_coverage_fraction=0.5,
    budget_replacement_margin=0.0,
    budget_coverage_loss_enabled=False,
    budget_coverage_min_loss_only=False,
    budget_coverage_joint_eviction_enabled=False,
    budget_patch_coverage_enabled=False,
    budget_patch_prototype_count=32,
    budget_patch_victim_selection_enabled=False,
    budget_patch_kcenter_victim_selection_enabled=False,
    budget_virtual_swap_enabled=False,
    budget_virtual_swap_max_normal_score_shift=0.05,
    budget_virtual_swap_max_iqr_increase=0.05,
    budget_virtual_swap_history_size=32,
    candidate_reliability_enabled=False,
    candidate_reliability_floor=0.1,
    candidate_unidirectional_weight=0.25,
    candidate_reliability_position_radius=1,
    candidate_patch_reliability_mode="legacy_rank_product",
    reference_reliability_penalty=1.0,
    budget_coverage_channel_weight=1.0,
    budget_coverage_lab_weight=1.0,
    budget_coverage_dino_weight=1.0,
    minimum_active_experts=0,
    lifecycle_mode="ttl_delete",
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
      if cold_start_mode not in {"legacy", "adaptive", "confidence"}:
            raise ValueError(
                "cold_start_mode must be legacy, adaptive, or confidence"
            )
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
      if cold_start_provisional_confirm_steps < 1:
            raise ValueError(
                "cold_start_provisional_confirm_steps must be positive"
            )
      if cold_start_provisional_max_wait_steps < cold_start_provisional_confirm_steps:
            raise ValueError(
                "cold_start_provisional_max_wait_steps must be at least "
                "cold_start_provisional_confirm_steps"
            )
      if not 0.0 <= cold_start_provisional_weight <= 1.0:
            raise ValueError(
                "cold_start_provisional_weight must be within [0, 1]"
            )
      if not 0.0 <= cold_start_provisional_min_channel_support <= 1.0:
            raise ValueError(
                "cold_start_provisional_min_channel_support must be within "
                "[0, 1]"
            )
      if cold_start_confidence_history < 2:
            raise ValueError(
                "cold_start_confidence_history must be at least 2"
            )
      if cold_start_reference_confidence_mode not in {
            "state",
            "prequential",
        }:
            raise ValueError(
                "cold_start_reference_confidence_mode must be state or "
                "prequential"
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
      if photometric_history_max < committee_cap:
            raise ValueError(
                "photometric_history_max must be at least committee_cap"
            )
      if photometric_state_registry_cap is None:
            photometric_state_registry_cap = 2 * committee_cap
      if photometric_state_registry_cap < committee_cap:
            raise ValueError(
                "photometric_state_registry_cap must be at least committee_cap"
            )
      if photometric_state_distribution_enabled is None:
            photometric_state_distribution_enabled = bool(
                photometric_adaptive_candidate_evidence
            )
      if photometric_temporary_state_enabled is None:
            photometric_temporary_state_enabled = bool(
                photometric_adaptive_candidate_evidence
            )
      if not 0.0 < photometric_denstream_decay <= 1.0:
            raise ValueError(
                "photometric_denstream_decay must be within (0, 1]"
            )
      if photometric_denstream_mature_support <= 0.0:
            raise ValueError(
                "photometric_denstream_mature_support must be positive"
            )
      if not (
            0.0 <= photometric_denstream_discard_support
            < photometric_denstream_mature_support
      ):
            raise ValueError(
                "photometric_denstream_discard_support must be non-negative "
                "and below photometric_denstream_mature_support"
            )
      if photometric_history_distance_threshold <= 0.0:
            raise ValueError(
                "photometric_history_distance_threshold must be positive"
            )
      if not 0.0 <= photometric_dino_similarity_floor <= 1.0:
            raise ValueError(
                "photometric_dino_similarity_floor must be within [0, 1]"
            )
      if photometric_history_reentry_members < 2:
            raise ValueError(
                "photometric_history_reentry_members must be at least 2"
            )
      if (
            photometric_history_reentry_members
            > photometric_dynamic_max_low_risk_members
      ):
            raise ValueError(
                "photometric_history_reentry_members must not exceed "
                "photometric_dynamic_max_low_risk_members"
            )
      if photometric_history_median_risk_max < 1.0:
            raise ValueError(
                "photometric_history_median_risk_max must be at least 1"
            )
      if (
            photometric_history_extreme_risk_max
            < photometric_history_median_risk_max
      ):
            raise ValueError(
                "photometric_history_extreme_risk_max must be at least "
                "photometric_history_median_risk_max"
            )
      if budget_coverage_window < 1:
            raise ValueError("budget_coverage_window must be at least 1")
      if not 0.0 <= budget_min_new_coverage_fraction <= 1.0:
            raise ValueError(
                "budget_min_new_coverage_fraction must be within [0, 1]"
            )
      if not 0.0 <= budget_replacement_margin <= 1.0:
            raise ValueError(
                "budget_replacement_margin must be within [0, 1]"
            )
      if budget_virtual_swap_max_normal_score_shift < 0.0:
            raise ValueError(
                "budget_virtual_swap_max_normal_score_shift must be non-negative"
            )
      if budget_virtual_swap_max_iqr_increase < 0.0:
            raise ValueError(
                "budget_virtual_swap_max_iqr_increase must be non-negative"
            )
      if budget_virtual_swap_history_size < 1:
            raise ValueError(
                "budget_virtual_swap_history_size must be at least 1"
            )
      coverage_weights = (
            budget_coverage_channel_weight,
            budget_coverage_lab_weight,
            budget_coverage_dino_weight,
      )
      if any(float(weight) < 0.0 for weight in coverage_weights):
            raise ValueError("budget coverage weights must be non-negative")
      if sum(float(weight) for weight in coverage_weights) <= 0.0:
            raise ValueError("at least one budget coverage weight must be positive")
      if photometric_budget_admission_mode not in {
            "channel_only",
            "channel_or_lab",
      }:
            raise ValueError(
                "photometric_budget_admission_mode must be channel_only "
                "or channel_or_lab"
            )
      if not 0.0 < candidate_reliability_floor <= 1.0:
            raise ValueError(
                "candidate_reliability_floor must be within (0, 1]"
            )
      if not 0.0 <= candidate_unidirectional_weight <= 1.0:
            raise ValueError(
                "candidate_unidirectional_weight must be within [0, 1]"
            )
      if candidate_reliability_position_radius < 0:
            raise ValueError(
                "candidate_reliability_position_radius must be non-negative"
            )
      if candidate_patch_reliability_mode not in {
            "legacy_rank_product",
            "cross_patch_robust",
      }:
            raise ValueError(
                "candidate_patch_reliability_mode must be "
                "legacy_rank_product or cross_patch_robust"
            )
      if reference_reliability_penalty < 0.0:
            raise ValueError(
                "reference_reliability_penalty must be non-negative"
            )
      if not 0 <= minimum_active_experts <= committee_cap:
            raise ValueError(
                "minimum_active_experts must be between 0 and committee_cap"
            )
      if lifecycle_mode not in {"ttl_delete", "dormant_replace"}:
            raise ValueError(
                "lifecycle_mode must be ttl_delete or dormant_replace"
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
      self.cold_start_provisional_enabled = bool(
            cold_start_provisional_enabled
        )
      self.cold_start_provisional_confirm_steps = int(
            cold_start_provisional_confirm_steps
        )
      self.cold_start_provisional_max_wait_steps = int(
            cold_start_provisional_max_wait_steps
        )
      self.cold_start_provisional_weight = float(
            cold_start_provisional_weight
        )
      self.cold_start_provisional_min_channel_support = float(
            cold_start_provisional_min_channel_support
        )
      self.cold_start_confidence_history = int(
            cold_start_confidence_history
        )
      self.cold_start_reference_confidence_mode = (
            cold_start_reference_confidence_mode
        )
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
      self.photometric_fitness_history_enabled = bool(
            photometric_fitness_history_enabled
        )
      self.photometric_history_min = int(photometric_history_min)
      self.photometric_history_max = int(photometric_history_max)
      self.photometric_history_distance_threshold = float(
            photometric_history_distance_threshold
      )
      self.photometric_dino_similarity_floor = float(
            photometric_dino_similarity_floor
      )
      self.photometric_history_reentry_enabled = bool(
            photometric_history_reentry_enabled
      )
      self.photometric_history_reentry_members = int(
            photometric_history_reentry_members
      )
      self.photometric_history_median_risk_max = float(
            photometric_history_median_risk_max
      )
      self.photometric_history_extreme_risk_max = float(
            photometric_history_extreme_risk_max
      )
      self.photometric_state_registry_cap = int(
            photometric_state_registry_cap
      )
      self.photometric_state_distribution_enabled = bool(
            photometric_state_distribution_enabled
      )
      self.photometric_temporary_state_enabled = bool(
            photometric_temporary_state_enabled
      )
      self.photometric_adaptive_candidate_evidence = bool(
            photometric_adaptive_candidate_evidence
        )
      self.photometric_stable_lab_anchor_enabled = bool(
            photometric_stable_lab_anchor_enabled
        )
      self.photometric_denstream_support_enabled = bool(
            photometric_denstream_support_enabled
        )
      self.photometric_denstream_decay = float(
            photometric_denstream_decay
        )
      self.photometric_denstream_mature_support = float(
            photometric_denstream_mature_support
        )
      self.photometric_denstream_discard_support = float(
            photometric_denstream_discard_support
        )
      self.photometric_budget_replacement = bool(
            photometric_budget_replacement
      )
      self.photometric_budget_admission_mode = str(
            photometric_budget_admission_mode
      )
      self.legacy_budget_replacement = bool(legacy_budget_replacement)
      self.budget_coverage_window = int(budget_coverage_window)
      self.budget_min_new_coverage_fraction = float(
            budget_min_new_coverage_fraction
      )
      self.budget_replacement_margin = float(budget_replacement_margin)
      self.budget_coverage_loss_enabled = bool(
            budget_coverage_loss_enabled
      )
      self.budget_coverage_min_loss_only = bool(
            budget_coverage_min_loss_only
      )
      self.budget_coverage_joint_eviction_enabled = bool(
            budget_coverage_joint_eviction_enabled
      )
      self.budget_patch_coverage_enabled = bool(
            budget_patch_coverage_enabled
      )
      self.budget_patch_prototype_count = int(
            budget_patch_prototype_count
      )
      if self.budget_patch_prototype_count <= 0:
            raise ValueError(
                "budget_patch_prototype_count must be positive"
            )
      self.budget_patch_victim_selection_enabled = bool(
            budget_patch_victim_selection_enabled
      )
      self.budget_patch_kcenter_victim_selection_enabled = bool(
            budget_patch_kcenter_victim_selection_enabled
      )
      self.budget_virtual_swap_enabled = bool(
            budget_virtual_swap_enabled
      )
      self.budget_virtual_swap_max_normal_score_shift = float(
            budget_virtual_swap_max_normal_score_shift
      )
      self.budget_virtual_swap_max_iqr_increase = float(
            budget_virtual_swap_max_iqr_increase
      )
      self.budget_virtual_swap_history_size = int(
            budget_virtual_swap_history_size
      )
      self.candidate_reliability_enabled = bool(
            candidate_reliability_enabled
      )
      self.candidate_reliability_floor = float(
            candidate_reliability_floor
      )
      self.candidate_unidirectional_weight = float(
            candidate_unidirectional_weight
      )
      self.candidate_reliability_position_radius = int(
            candidate_reliability_position_radius
      )
      self.candidate_patch_reliability_mode = str(
            candidate_patch_reliability_mode
      )
      self.reference_reliability_penalty = float(
            reference_reliability_penalty
      )
      self.budget_coverage_channel_weight = float(
            budget_coverage_channel_weight
      )
      self.budget_coverage_lab_weight = float(
            budget_coverage_lab_weight
      )
      self.budget_coverage_dino_weight = float(
            budget_coverage_dino_weight
      )
      self.minimum_active_experts = int(minimum_active_experts)
      self.lifecycle_mode = lifecycle_mode
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
      self.last_fitness_history_profile = None
      self.confirmed_provisional = None
      self.bootstrap_members = []
      self.bootstrap_provisional = None
      self.bootstrap_confidence_members = []
      self.bootstrap_confidence_expert_id = None
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
      self.photometric_handled_lab_center = None
      self.photometric_rearm_streak = 0
      self.photometric_pool_abort_reason = None
      self.historical_states = []
      self.next_historical_state_id = 0
      self.state_lifecycle_events = []
      self.last_budget_replacement = None
      self.last_budget_replacement_evaluation = None
      self.last_history_reentry_event = None
      self.coverage_history = []

    def provisional_expert_before_step(self):
        if self.bootstrap_provisional is not None:
            representatives = self.bootstrap_provisional["members"]
            representative = representatives[0]
            return {
                "image_id": int(representative["image_id"]),
                "image_ids": [
                    int(member["image_id"]) for member in representatives
                ],
                "memory_members": [],
                "representative_step": int(representative["step"]),
                "start_step": int(self.bootstrap_provisional["start_step"]),
                "pool_size": len(representatives),
                "weight": float(self.cold_start_provisional_weight),
                "state": "bootstrap_provisional",
            }
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
            if expert.get("lifecycle_state", "active") != "active":
                continue
            snapshot = {
                "expert_id":expert["expert_id"],
                "image_id":expert["image_id"],
                "cluster_id":expert["cluster_id"],
                "admission_step":expert["admission_step"],
            }
            if expert.get("admission_route", "legacy") != "legacy":
                snapshot["admission_route"] = expert["admission_route"]
            if expert.get("scoring_weight") is not None:
                snapshot["scoring_weight"] = float(
                    expert["scoring_weight"]
                )
                snapshot["state_confidence"] = float(
                    expert.get("state_confidence", expert["scoring_weight"])
                )
                snapshot["normality_confidence"] = float(
                    expert.get("normality_confidence", 1.0)
                )
                snapshot["reference_confidence"] = float(
                    expert.get(
                        "reference_confidence", expert["scoring_weight"]
                    )
                )
            if expert.get("state_effective_support") is not None:
                snapshot["state_effective_support"] = float(
                    expert["state_effective_support"]
                )
                snapshot["state_member_image_ids"] = [
                    int(member_id)
                    for member_id in expert.get(
                        "state_member_image_ids", [expert["image_id"]]
                    )
                ]
                snapshot["state_member_weights"] = [
                    float(weight)
                    for weight in expert.get("state_member_weights", [])
                ]
            if expert.get("scoring_patch_reliabilities") is not None:
                snapshot["scoring_patch_reliabilities"] = list(
                    expert["scoring_patch_reliabilities"]
                )
                snapshot["member_reliability"] = float(
                    expert.get("member_reliability", 1.0)
                )
                snapshot["reference_reliability_penalty"] = float(
                    expert.get("reference_reliability_penalty", 1.0)
                )
                snapshot["reference_member_image_ids"] = [
                    int(value)
                    for value in expert.get(
                        "reference_member_image_ids",
                        [expert["image_id"]],
                    )
                ]
                snapshot["reference_member_reliabilities"] = [
                    float(value)
                    for value in expert.get(
                        "reference_member_reliabilities",
                        [expert.get("member_reliability", 1.0)],
                    )
                ]
            if expert.get("lab_state") is not None:
                snapshot["lab_state"] = dict(expert["lab_state"])
            snapshots.append(snapshot)
        return snapshots

    def expert_slots_before_step(self):
        snapshots = []
        for expert in self.active_experts:
            snapshot = {
                "expert_id": expert["expert_id"],
                "image_id": expert["image_id"],
                "cluster_id": expert["cluster_id"],
                "admission_step": expert["admission_step"],
                "lifecycle_state": expert.get(
                    "lifecycle_state", "active"
                ),
            }
            if expert.get("admission_route", "legacy") != "legacy":
                snapshot["admission_route"] = expert["admission_route"]
            if expert.get("scoring_weight") is not None:
                snapshot["scoring_weight"] = float(
                    expert["scoring_weight"]
                )
                snapshot["state_confidence"] = float(
                    expert.get("state_confidence", expert["scoring_weight"])
                )
                snapshot["normality_confidence"] = float(
                    expert.get("normality_confidence", 1.0)
                )
                snapshot["reference_confidence"] = float(
                    expert.get(
                        "reference_confidence", expert["scoring_weight"]
                    )
                )
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
                "scoring_weight": expert.get("scoring_weight"),
                "state_confidence": expert.get("state_confidence"),
                "normality_confidence": expert.get(
                    "normality_confidence"
                ),
                "reference_confidence": expert.get(
                    "reference_confidence"
                ),
                "state_effective_support": expert.get(
                    "state_effective_support"
                ),
                "state_member_image_ids": [
                    int(member_id)
                    for member_id in expert.get(
                        "state_member_image_ids", []
                    )
                ],
                "state_member_weights": [
                    float(weight)
                    for weight in expert.get("state_member_weights", [])
                ],
                "member_reliability": expert.get("member_reliability"),
                "scoring_patch_reliability_mean": (
                    float(
                        np.mean(expert["scoring_patch_reliabilities"])
                    )
                    if expert.get("scoring_patch_reliabilities")
                    else None
                ),
                "reliability_audit": expert.get("reliability_audit"),
                "lifecycle_state": expert.get(
                    "lifecycle_state", "active"
                ),
            }
            for expert in self.active_experts
        ]

    def fuser_training_members(self, step):
        members = []
        active_cluster_ids = {
            expert["cluster_id"]
            for expert in self.active_experts
            if expert.get("lifecycle_state", "active") == "active"
        }

        for cluster_id in active_cluster_ids:
            for member in self.clusters[cluster_id]["members"]:
                if member["step"] < step:
                    members.append(
                        {
                            "step": member["step"],
                            "image_id": member["image_id"],
                            "cluster_id": cluster_id,
                        **(
                            {
                                "patch_reliabilities": [
                                    float(value)
                                    for value in torch.as_tensor(
                                        member["patch_reliabilities"],
                                        dtype=torch.float32,
                                    ).tolist()
                                ],
                                "scoring_patch_reliabilities": [
                                    float(value)
                                    for value in torch.as_tensor(
                                        member.get(
                                            "scoring_patch_reliabilities",
                                            member["patch_reliabilities"],
                                        ),
                                        dtype=torch.float32,
                                    ).tolist()
                                ],
                            }
                            if member.get("patch_reliabilities") is not None
                            else {}
                        ),
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

    def _patch_coreset(
        self,
        dino_patch_features,
        patch_ids=None,
        patch_reliabilities=None,
    ):
        if not self.budget_patch_coverage_enabled:
            return None
        features = dino_patch_features.detach().float().cpu()
        if features.ndim != 2:
            raise ValueError(
                "dino_patch_features must have shape "
                "[patch_count, feature_dim]"
            )
        reliabilities = None
        if patch_reliabilities is not None:
            reliabilities = torch.as_tensor(
                patch_reliabilities,
                dtype=torch.float32,
            ).detach().cpu()
            if reliabilities.shape != (features.shape[0],):
                raise ValueError(
                    "patch_reliabilities must contain one value per patch"
                )
            if ((reliabilities <= 0.0) | (reliabilities > 1.0)).any():
                raise ValueError(
                    "patch_reliabilities must be within (0, 1]"
                )
        if patch_ids:
            indices = torch.as_tensor(
                sorted({int(patch_id) for patch_id in patch_ids}),
                dtype=torch.long,
            )
            features = features.index_select(0, indices)
            if reliabilities is not None:
                reliabilities = reliabilities.index_select(0, indices)
        features = F.normalize(features, dim=1)
        budget = min(
            int(self.budget_patch_prototype_count),
            int(features.shape[0]),
        )
        if budget == int(features.shape[0]):
            return features
        center = F.normalize(
            (
                features * reliabilities[:, None]
                if reliabilities is not None
                else features
            ).sum(dim=0),
            dim=0,
        )
        first_scores = features @ center
        if reliabilities is not None:
            first_scores = first_scores * reliabilities
        first = int(first_scores.argmax())
        selected = [first]
        maximum_similarity = features @ features[first]
        for _ in range(1, budget):
            selection_score = maximum_similarity
            if reliabilities is not None:
                selection_score = selection_score + (1.0 - reliabilities)
            next_index = int(selection_score.argmin())
            selected.append(next_index)
            maximum_similarity = torch.maximum(
                maximum_similarity,
                features @ features[next_index],
            )
        return features.index_select(
            0,
            torch.as_tensor(selected, dtype=torch.long),
        )

    def _candidate_patch_bank(
        self,
        dino_patch_features,
        patch_reliabilities,
    ):
        if not self.candidate_reliability_enabled:
            return None
        features = F.normalize(
            dino_patch_features.detach().float().cpu(),
            dim=1,
        )
        reliabilities = torch.as_tensor(
            patch_reliabilities,
            dtype=torch.float32,
        ).detach().cpu()
        if reliabilities.shape != (features.shape[0],):
            raise ValueError(
                "patch_reliabilities must contain one value per patch"
            )
        if not torch.isfinite(reliabilities).all():
            raise ValueError("patch_reliabilities contain non-finite values")
        if ((reliabilities <= 0.0) | (reliabilities > 1.0)).any():
            raise ValueError("patch_reliabilities must be within (0, 1]")
        patch_ids = torch.arange(features.shape[0], dtype=torch.long)
        return {
            "features": features,
            "patch_ids": patch_ids,
            "base_reliabilities": reliabilities,
            "patch_count": int(features.shape[0]),
        }

    def _attach_candidate_reliability(
        self,
        member,
        dino_patch_features,
        patch_reliabilities,
    ):
        if (
            not self.candidate_reliability_enabled
            or self.candidate_patch_reliability_mode
            != "cross_patch_robust"
        ):
            return
        if patch_reliabilities is None:
            patch_reliabilities = torch.ones(
                dino_patch_features.shape[0],
                dtype=torch.float32,
            )
        base = torch.as_tensor(
            patch_reliabilities,
            dtype=torch.float32,
        ).detach().cpu()
        if base.shape != (dino_patch_features.shape[0],):
            raise ValueError(
                "patch_reliabilities must contain one value per patch"
            )
        if not torch.isfinite(base).all():
            raise ValueError("patch_reliabilities contain non-finite values")
        if ((base <= 0.0) | (base > 1.0)).any():
            raise ValueError("patch_reliabilities must be within (0, 1]")
        member["base_patch_reliabilities"] = base
        member["patch_reliabilities"] = base.clone()
        member["scoring_patch_reliabilities"] = torch.ones_like(base)
        member["reliability_patch_bank"] = self._candidate_patch_bank(
            dino_patch_features,
            base,
        )
        member["member_reliability"] = 0.5

    def _bidirectional_member_support(
        self,
        left,
        right,
        *,
        return_patch_support=False,
    ):
        left_bank = left.get("reliability_patch_bank")
        right_bank = right.get("reliability_patch_bank")
        if left_bank is None or right_bank is None:
            audit = {
                "mutual_fraction": 0.0,
                "left_directional_support": 0.5,
                "right_directional_support": 0.5,
            }
            if return_patch_support:
                return 0.5, 0.5, None, None, audit
            return 0.5, 0.5, audit
        left_features = left_bank["features"]
        right_features = right_bank["features"]
        left_grid = math.isqrt(int(left_bank["patch_count"]))
        right_grid = math.isqrt(int(right_bank["patch_count"]))
        if (
            left_grid * left_grid != int(left_bank["patch_count"])
            or right_grid != left_grid
        ):
            similarities = (left_features @ right_features.T).clamp(
                0.0, 1.0
            )
            left_values, left_matches = similarities.max(dim=1)
            right_values, right_matches = similarities.max(dim=0)
        else:
            radius = self.candidate_reliability_position_radius
            patch_count = int(left_bank["patch_count"])
            left_values = torch.full((patch_count,), -1.0)
            right_values = torch.full((patch_count,), -1.0)
            left_matches = torch.full((patch_count,), -1, dtype=torch.long)
            right_matches = torch.full((patch_count,), -1, dtype=torch.long)
            patch_grid = torch.arange(patch_count).reshape(
                left_grid, left_grid
            )
            for row_delta in range(-radius, radius + 1):
                for col_delta in range(-radius, radius + 1):
                    left_row_start = max(0, -row_delta)
                    left_row_end = left_grid - max(0, row_delta)
                    left_col_start = max(0, -col_delta)
                    left_col_end = left_grid - max(0, col_delta)
                    right_row_start = left_row_start + row_delta
                    right_row_end = left_row_end + row_delta
                    right_col_start = left_col_start + col_delta
                    right_col_end = left_col_end + col_delta
                    left_indices = patch_grid[
                        left_row_start:left_row_end,
                        left_col_start:left_col_end,
                    ].flatten()
                    right_indices = patch_grid[
                        right_row_start:right_row_end,
                        right_col_start:right_col_end,
                    ].flatten()
                    values = (
                        left_features.index_select(0, left_indices)
                        * right_features.index_select(0, right_indices)
                    ).sum(dim=1).clamp(0.0, 1.0)
                    left_better = values > left_values[left_indices]
                    left_values[left_indices[left_better]] = values[left_better]
                    left_matches[left_indices[left_better]] = right_indices[
                        left_better
                    ]
                    right_better = values > right_values[right_indices]
                    right_values[right_indices[right_better]] = values[
                        right_better
                    ]
                    right_matches[right_indices[right_better]] = left_indices[
                        right_better
                    ]
        left_available = left_values >= 0.0
        right_available = right_values >= 0.0
        left_mutual = torch.zeros_like(left_available)
        if left_available.any():
            left_indices = torch.arange(left_values.numel())
            left_mutual[left_available] = (
                right_matches[left_matches[left_available]]
                == left_indices[left_available]
            )
        right_mutual = torch.zeros_like(right_available)
        if right_available.any():
            right_indices = torch.arange(right_values.numel())
            right_mutual[right_available] = (
                left_matches[right_matches[right_available]]
                == right_indices[right_available]
            )
        weak = self.candidate_unidirectional_weight
        left_weights = torch.where(left_mutual, 1.0, weak)
        right_weights = torch.where(right_mutual, 1.0, weak)
        floor = self.candidate_reliability_floor
        left_patch_support = torch.full_like(left_values, floor)
        right_patch_support = torch.full_like(right_values, floor)
        left_weighted_support = (
            left_values[left_available] * left_weights[left_available]
        )
        right_weighted_support = (
            right_values[right_available] * right_weights[right_available]
        )
        left_patch_support[left_available] = left_weighted_support.clamp(
            min=floor,
            max=1.0,
        )
        right_patch_support[right_available] = right_weighted_support.clamp(
            min=floor,
            max=1.0,
        )
        left_support = (
            float(left_weighted_support.mean())
            if left_available.any()
            else floor
        )
        right_support = (
            float(right_weighted_support.mean())
            if right_available.any()
            else floor
        )
        mutual_count = int(left_mutual.sum())
        available_count = int(left_available.sum())
        audit = {
            "mutual_fraction": (
                float(mutual_count / available_count)
                if available_count
                else 0.0
            ),
            "left_directional_support": left_support,
            "right_directional_support": right_support,
        }
        if return_patch_support:
            return (
                left_support,
                right_support,
                left_patch_support,
                right_patch_support,
                audit,
            )
        return left_support, right_support, audit

    def _cross_patch_scoring_reliability(self, patch_support):
        support = torch.as_tensor(
            patch_support,
            dtype=torch.float32,
        ).detach().cpu().clamp(
            min=self.candidate_reliability_floor,
            max=1.0,
        )
        center = support.median()
        mad = (support - center).abs().median()
        scale = max(float(1.4826 * mad), 0.02)
        low_outlier_strength = (
            (center - support) / scale - 2.5
        ).clamp_min(0.0)
        return (1.0 / (1.0 + low_outlier_strength)).clamp(
            min=self.candidate_reliability_floor,
            max=1.0,
        )

    def _refresh_candidate_reliabilities(self, members):
        if not self.candidate_reliability_enabled or not members:
            return
        support_values = [[] for _ in members]
        patch_support_values = [[] for _ in members]
        pair_audits = [[] for _ in members]
        for left_index, right_index in itertools.combinations(
            range(len(members)), 2
        ):
            (
                left_support,
                right_support,
                left_patch_support,
                right_patch_support,
                pair_audit,
            ) = (
                self._bidirectional_member_support(
                    members[left_index],
                    members[right_index],
                    return_patch_support=True,
                )
            )
            support_values[left_index].append(left_support)
            support_values[right_index].append(right_support)
            if left_patch_support is not None:
                patch_support_values[left_index].append(left_patch_support)
            if right_patch_support is not None:
                patch_support_values[right_index].append(right_patch_support)
            pair_audits[left_index].append(pair_audit)
            pair_audits[right_index].append(pair_audit)

        floor = self.candidate_reliability_floor
        for index, member in enumerate(members):
            base = member.get("base_patch_reliabilities")
            if base is None:
                continue
            base = torch.as_tensor(base, dtype=torch.float32).cpu()
            if self.candidate_patch_reliability_mode == "cross_patch_robust":
                channel_evidence = member.get("channel_support")
                if channel_evidence is None:
                    channel_evidence = member.get(
                        "provisional_channel_support"
                    )
                channel_evidence = (
                    float(channel_evidence)
                    if channel_evidence is not None
                    else 0.5
                )
            else:
                channel_evidence = float(base.mean())
            threshold = member.get("historical_ms_threshold")
            score = member.get("ms_score")
            risk_evidence = 0.5
            if (
                threshold is not None
                and score is not None
                and float(threshold) > 0.0
            ):
                risk_ratio = float(score) / max(float(threshold), 1e-12)
                risk_evidence = min(1.0, 1.0 / max(risk_ratio, 1e-12))
            cross_support = (
                float(np.mean(support_values[index]))
                if support_values[index]
                else 0.5
            )
            member_reliability = float(
                max(
                    floor,
                    min(
                        1.0,
                        (
                            max(channel_evidence, floor)
                            * max(risk_evidence, floor)
                            * max(cross_support, floor)
                        )
                        ** (1.0 / 3.0),
                    ),
                )
            )
            effective = (base * member_reliability).clamp(
                min=floor,
                max=1.0,
            )
            if self.candidate_patch_reliability_mode == "cross_patch_robust":
                if patch_support_values[index]:
                    cross_patch_support = torch.stack(
                        patch_support_values[index],
                        dim=0,
                    ).mean(dim=0)
                    scoring_reliability = (
                        self._cross_patch_scoring_reliability(
                            cross_patch_support
                        )
                    )
                else:
                    cross_patch_support = torch.ones_like(base)
                    scoring_reliability = torch.ones_like(base)
            else:
                cross_patch_support = None
                scoring_reliability = effective
            member["member_reliability"] = member_reliability
            member["patch_reliabilities"] = effective
            member["scoring_patch_reliabilities"] = scoring_reliability
            patch_bank = member.get("reliability_patch_bank")
            if patch_bank is not None:
                member["patch_prototypes"] = self._patch_coreset(
                    patch_bank["features"],
                    patch_reliabilities=effective,
                )
            member["reliability_audit"] = {
                "channel_evidence": channel_evidence,
                "risk_evidence": risk_evidence,
                "bidirectional_support": cross_support,
                "member_reliability": member_reliability,
                "patch_reliability_mean": float(effective.mean()),
                "patch_reliability_min": float(effective.min()),
                "scoring_patch_reliability_mean": float(
                    scoring_reliability.mean()
                ),
                "scoring_patch_reliability_min": float(
                    scoring_reliability.min()
                ),
                "cross_patch_support_mean": (
                    float(cross_patch_support.mean())
                    if cross_patch_support is not None
                    else None
                ),
                "cross_patch_support_min": (
                    float(cross_patch_support.min())
                    if cross_patch_support is not None
                    else None
                ),
                "patch_reliability_mode": (
                    self.candidate_patch_reliability_mode
                ),
                "pair_support": pair_audits[index],
            }

    def _candidate_weighted_embedding_sum(self, members):
        if not self.candidate_reliability_enabled:
            return torch.stack(
                [member["embedding"] for member in members]
            ).sum(dim=0)
        self._refresh_candidate_reliabilities(members)
        return torch.stack(
            [
                member["embedding"]
                * float(member.get("member_reliability", 1.0))
                for member in members
            ]
        ).sum(dim=0)

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

    def _build_lab_state(
        self,
        members,
        existing_state=None,
        freeze_anchor=True,
        step=None,
    ):
        eligible_members = [
            member
            for member in members
            if member.get("lab_feature") is not None
        ]
        features = [member["lab_feature"] for member in eligible_members]
        if not self.cold_start_lab_enabled or len(features) != len(members):
            return None

        features = np.stack(features, axis=0)
        if self.candidate_reliability_enabled:
            weights = np.asarray(
                [
                    float(member.get("member_reliability", 1.0))
                    for member in eligible_members
                ],
                dtype=np.float64,
            )
            center = np.average(features, axis=0, weights=weights)
        else:
            center = np.median(features, axis=0)
        profiles = [
            self._lab_distances(feature, center)
            for feature in features
        ]
        state = {
            "member_image_ids": [int(member["image_id"]) for member in members],
            "center": center.tolist(),
            "coverage_center": center.tolist(),
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
            "detector_center": center.tolist(),
            "detector_ready": bool(freeze_anchor),
            "detector_rebase_count": 0,
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

        existing_state = existing_state or {}
        if existing_state.get("detector_ready", False):
            state["detector_center"] = list(
                existing_state.get(
                    "detector_center",
                    existing_state.get(
                        "anchor_center",
                        existing_state.get("center", center.tolist()),
                    ),
                )
            )
            state["detector_ready"] = True
            state["detector_rebase_count"] = int(
                existing_state.get("detector_rebase_count", 0)
            )
            state["recent_lab_history"] = list(
                existing_state.get("recent_lab_history", [])
            )
            state["candidate_lab_history"] = list(
                existing_state.get("candidate_lab_history", [])
            )
            for channel in ("l", "ab"):
                for suffix in (
                    "baseline_median",
                    "baseline_mad",
                    "threshold",
                    "ewma",
                    "streak",
                ):
                    key = f"{channel}_{suffix}"
                    if key in existing_state:
                        state[key] = existing_state[key]
            state["drift_confirmed"] = bool(
                existing_state.get("drift_confirmed", False)
            )

        if not self.photometric_stable_lab_anchor_enabled:
            return state

        anchor_frozen = bool(existing_state.get("anchor_frozen", False))
        if anchor_frozen:
            state["anchor_center"] = list(existing_state["anchor_center"])
            state["anchor_member_image_ids"] = list(
                existing_state.get("anchor_member_image_ids", [])
            )
            state["anchor_created_step"] = existing_state.get(
                "anchor_created_step"
            )
            state["anchor_frozen"] = True
            state["recent_lab_history"] = list(
                existing_state.get("recent_lab_history", [])
            )
            state["candidate_lab_history"] = list(
                existing_state.get("candidate_lab_history", [])
            )
            for channel in ("l", "ab"):
                for suffix in (
                    "baseline_median",
                    "baseline_mad",
                    "threshold",
                    "ewma",
                    "streak",
                ):
                    state[f"{channel}_{suffix}"] = existing_state[
                        f"{channel}_{suffix}"
                    ]
            state["drift_confirmed"] = bool(
                existing_state.get("drift_confirmed", False)
            )
            return state

        state["anchor_center"] = center.tolist()
        state["anchor_member_image_ids"] = [
            int(member["image_id"]) for member in members
        ]
        state["anchor_created_step"] = (
            int(step) if step is not None else None
        )
        state["anchor_frozen"] = bool(freeze_anchor)
        state["candidate_lab_history"] = []
        return state

    def _rebase_lab_detector(self, expert, members, step):
        state = expert.get("lab_state")
        features = [
            member.get("lab_feature")
            for member in members
            if member.get("lab_feature") is not None
        ]
        if state is None or not features:
            return False

        features = np.stack(features, axis=0)
        center = np.median(features, axis=0)
        profiles = [
            self._lab_distances(feature, center) for feature in features
        ]
        state["detector_center"] = center.tolist()
        state["detector_ready"] = True
        state["detector_rebase_count"] = int(
            state.get("detector_rebase_count", 0)
        ) + 1
        state["detector_rebase_step"] = int(step)
        state["drift_confirmed"] = False
        state["candidate_lab_history"] = []
        state["recent_lab_history"] = [
            {
                "step": int(member["step"]),
                "image_id": int(member["image_id"]),
                "feature": np.asarray(
                    member["lab_feature"], dtype=np.float64
                ).tolist(),
            }
            for member in members
            if member.get("lab_feature") is not None
        ][-self.photometric_history_max:]
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
            state[f"{channel}_streak"] = 0
        return True

    def _rebase_photometric_trigger_detector(self, members, step):
        expert_id = (
            self.photometric_trigger_expert_id
            if self.photometric_trigger_expert_id is not None
            else self.photometric_episode_reference_expert_id
        )
        expert = next(
            (
                item
                for item in self.active_experts
                if expert_id is not None
                and int(item["expert_id"]) == int(expert_id)
            ),
            None,
        )
        if expert is None:
            return False
        return self._rebase_lab_detector(expert, members, step)

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

    def _admit_bootstrap_expert(
        self,
        selected,
        similarities,
        step,
        pool_size,
        minimum_similarity,
        *,
        forced,
        confirmation_steps=0,
    ):
        self._refresh_candidate_reliabilities(selected)
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
            "lifecycle_state": "active",
            "last_channel_support": None,
            "representative_mode": "bootstrap_medoid",
            "admission_ttl_mode": "base",
            "admission_route": "bootstrap",
            "lab_state": self._build_lab_state(selected, step=step),
        }
        if (
            self.candidate_reliability_enabled
            and self.candidate_patch_reliability_mode
            == "cross_patch_robust"
        ):
            expert["member_reliability"] = float(
                representative.get("member_reliability", 1.0)
            )
            expert["scoring_patch_reliabilities"] = [
                float(value)
                for value in torch.as_tensor(
                    representative.get(
                        "scoring_patch_reliabilities",
                        representative["patch_reliabilities"],
                    ),
                    dtype=torch.float32,
                ).tolist()
            ]
            expert["reference_reliability_penalty"] = float(
                self.reference_reliability_penalty
            )
            expert["reliability_audit"] = dict(
                representative.get("reliability_audit", {})
            )
            expert["reference_member_image_ids"] = [
                int(item["image_id"]) for item in selected
            ]
            expert["reference_member_reliabilities"] = [
                float(item.get("member_reliability", 1.0))
                for item in selected
            ]
        self.active_experts.append(expert)
        self._register_expert_historical_state(expert, "active", step)
        self.admitted_count += 1
        self.next_expert_id += 1
        self.bootstrap_provisional = None
        self.last_bootstrap_event = {
            "decision": "admitted",
            "pool_size": int(pool_size),
            "selected_image_ids": [
                int(item["image_id"]) for item in selected
            ],
            "representative_image_id": int(representative["image_id"]),
            "minimum_similarity": float(minimum_similarity),
            "forced": bool(forced),
            "confirmation_steps": int(confirmation_steps),
        }
        return True, cluster_id, expert["expert_id"], "bootstrap"

    @staticmethod
    def _continuous_rank(values, higher_is_better=True):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        ranks = np.full(values.shape, 0.5, dtype=np.float64)
        finite_values = values[finite]
        if finite_values.size == 0:
            return ranks
        for index in np.flatnonzero(finite):
            value = values[index]
            lower = float(np.sum(finite_values < value))
            equal = float(np.sum(finite_values == value))
            rank = (lower + 0.5 * equal) / finite_values.size
            ranks[index] = rank if higher_is_better else 1.0 - rank
        return ranks

    @staticmethod
    def _prequential_rank(value, history, higher_is_better):
        if value is None or not np.isfinite(value):
            return None
        history = np.asarray(history, dtype=np.float64)
        history = history[np.isfinite(history)]
        if history.size == 0:
            return None
        if higher_is_better:
            supported = np.sum(history <= float(value))
        else:
            supported = np.sum(history >= float(value))
        return float((1.0 + supported) / (history.size + 1.0))

    def _bootstrap_prequential_profile(self, prior_members, member):
        structural_support = None
        if len(prior_members) >= 2:
            prior_embeddings = torch.stack(
                [item["embedding"] for item in prior_members]
            )
            prior_similarities = prior_embeddings @ prior_embeddings.T
            prior_mask = ~torch.eye(
                len(prior_members),
                dtype=torch.bool,
                device=prior_similarities.device,
            )
            prior_centralities = prior_similarities[prior_mask].reshape(
                len(prior_members), len(prior_members) - 1
            ).median(dim=1).values
            current_similarities = (
                prior_embeddings @ member["embedding"]
            )
            current_centrality = float(
                current_similarities.median().item()
            )
            structural_support = self._prequential_rank(
                current_centrality,
                prior_centralities.detach().cpu().numpy(),
                higher_is_better=True,
            )

        risk_support = self._prequential_rank(
            member.get("ms_score"),
            [item.get("ms_score") for item in prior_members],
            higher_is_better=False,
        )
        channel_support = self._prequential_rank(
            member.get("channel_support"),
            [item.get("channel_support") for item in prior_members],
            higher_is_better=True,
        )
        supports = [
            value
            for value in (
                structural_support,
                risk_support,
                channel_support,
            )
            if value is not None
        ]
        combined_support = (
            float(np.prod(supports) ** (1.0 / len(supports)))
            if supports
            else None
        )
        return {
            "structural_support": structural_support,
            "risk_support": risk_support,
            "channel_support": channel_support,
            "combined_support": combined_support,
            "evidence_count": len(supports),
        }

    def _bootstrap_confidence_profile(self, members):
        embeddings = torch.stack([member["embedding"] for member in members])
        similarities = embeddings @ embeddings.T
        count = len(members)
        if count == 1:
            structural_centrality = np.asarray([1.0], dtype=np.float64)
        else:
            mask = ~torch.eye(
                count,
                dtype=torch.bool,
                device=similarities.device,
            )
            structural_centrality = similarities[mask].reshape(
                count, count - 1
            ).median(dim=1).values.cpu().numpy()

        structural_rank = self._continuous_rank(
            structural_centrality,
            higher_is_better=True,
        )
        risk_rank = self._continuous_rank(
            [
                member["ms_score"]
                if member.get("ms_score") is not None
                else np.nan
                for member in members
            ],
            higher_is_better=False,
        )
        support_rank = self._continuous_rank(
            [
                member["channel_support"]
                if member.get("channel_support") is not None
                else np.nan
                for member in members
            ],
            higher_is_better=True,
        )
        evidence = np.cbrt(
            np.maximum(structural_rank, 1e-12)
            * np.maximum(risk_rank, 1e-12)
            * np.maximum(support_rank, 1e-12)
        )
        if self.cold_start_reference_confidence_mode == "prequential":
            prequential_weights = np.asarray(
                [
                    member.get("prequential_support")
                    if member.get("prequential_support") is not None
                    else 0.5
                    for member in members
                ],
                dtype=np.float64,
            )
            evidence *= np.clip(prequential_weights, 1e-6, 1.0)
        weights = evidence / evidence.sum()
        weight_tensor = torch.as_tensor(
            weights,
            dtype=similarities.dtype,
            device=similarities.device,
        )
        medoid_scores = similarities @ weight_tensor
        representative_index = int(medoid_scores.argmax())
        effective_support = float(1.0 / np.square(weights).sum())

        if count == 1:
            coherence = 1.0
        else:
            pair_weights = np.outer(weights, weights)
            np.fill_diagonal(pair_weights, 0.0)
            pair_total = float(pair_weights.sum())
            pair_similarities = similarities.detach().cpu().numpy()
            coherence = float(
                (pair_weights * ((pair_similarities + 1.0) * 0.5)).sum()
                / max(pair_total, 1e-12)
            )
        state_confidence = float(
            np.clip(
                (1.0 - 1.0 / effective_support) * coherence,
                0.0,
                1.0,
            )
        )
        prequential_supports = np.asarray(
            [
                member.get("prequential_support", np.nan)
                if member.get("prequential_support") is not None
                else np.nan
                for member in members
            ],
            dtype=np.float64,
        )
        finite_prequential = prequential_supports[
            np.isfinite(prequential_supports)
        ]
        if self.cold_start_reference_confidence_mode == "prequential":
            normality_confidence = (
                float(finite_prequential.mean())
                if finite_prequential.size
                else 0.0
            )
            reference_confidence = float(
                state_confidence * normality_confidence
            )
        else:
            normality_confidence = 1.0
            reference_confidence = state_confidence
        return {
            "weights": weights,
            "similarities": similarities,
            "representative_index": representative_index,
            "effective_support": effective_support,
            "coherence": coherence,
            "confidence": state_confidence,
            "state_confidence": state_confidence,
            "normality_confidence": normality_confidence,
            "reference_confidence": reference_confidence,
            "prequential_support_count": int(finite_prequential.size),
            "structural_centrality": structural_centrality,
        }

    def _advance_confidence_cold_start(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        lab_feature,
        patch_reliabilities=None,
    ):
        member = {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": self._image_embedding(dino_patch_features),
            "patch_prototypes": self._patch_coreset(
                dino_patch_features
            ),
            "quality": float(channel_support or 0.0) - float(ms_score or 0.0),
            "ms_score": ms_score,
            "channel_support": channel_support,
            "ms_threshold": None,
            "support_threshold": None,
            "candidate_route": "bootstrap_confidence",
            "provisional_channel_support": None,
            "lab_feature": self._lab_feature(lab_feature),
        }
        self._attach_candidate_reliability(
            member,
            dino_patch_features,
            patch_reliabilities,
        )
        prequential_profile = None
        if self.cold_start_reference_confidence_mode == "prequential":
            prequential_profile = self._bootstrap_prequential_profile(
                self.bootstrap_confidence_members,
                member,
            )
            member["prequential_support"] = prequential_profile[
                "combined_support"
            ]
            member["prequential_profile"] = prequential_profile
        self.bootstrap_confidence_members.append(member)
        del self.bootstrap_confidence_members[
            :-self.cold_start_confidence_history
        ]
        self._refresh_candidate_reliabilities(
            self.bootstrap_confidence_members
        )
        profile = self._bootstrap_confidence_profile(
            self.bootstrap_confidence_members
        )
        representative = self.bootstrap_confidence_members[
            profile["representative_index"]
        ]
        member_ids = [
            int(item["image_id"])
            for item in self.bootstrap_confidence_members
        ]

        expert = next(
            (
                item
                for item in self.active_experts
                if item.get("expert_id")
                == self.bootstrap_confidence_expert_id
            ),
            None,
        )
        created = expert is None
        if created:
            cluster_id = len(self.clusters)
            expert = {
                "expert_id": self.next_expert_id,
                "image_id": int(representative["image_id"]),
                "cluster_id": cluster_id,
                "admission_step": int(step),
                "cluster_support_at_admission": 1,
                "max_support_gap": 0,
                "last_supported_step": int(step),
                "support_count": 1,
                "patience": self.base_ttl,
                "ttl": self.base_ttl,
                "deletion_step": None,
                "last_event": "confidence_state_created",
                "lifecycle_state": "active",
                "last_channel_support": channel_support,
                "representative_mode": "confidence_medoid",
                "admission_ttl_mode": "confidence_managed",
                "admission_route": "bootstrap_confidence",
                "lab_state": None,
            }
            self.clusters.append(
                {
                    "embedding_sum": representative["embedding"].clone(),
                    "members": list(self.bootstrap_confidence_members),
                    "active_expert_id": expert["expert_id"],
                }
            )
            self.active_experts.append(expert)
            self.bootstrap_confidence_expert_id = int(expert["expert_id"])
            self.admitted_count += 1
            self.next_expert_id += 1
        else:
            cluster_id = int(expert["cluster_id"])

        weights = profile["weights"]
        weighted_embedding = torch.stack(
            [
                item["embedding"] * float(weight)
                for item, weight in zip(
                    self.bootstrap_confidence_members, weights
                )
            ]
        ).sum(dim=0)
        self.clusters[cluster_id]["embedding_sum"] = weighted_embedding
        self.clusters[cluster_id]["members"] = list(
            self.bootstrap_confidence_members
        )
        expert["image_id"] = int(representative["image_id"])
        expert["last_supported_step"] = int(step)
        expert["support_count"] = len(self.bootstrap_confidence_members)
        expert["last_event"] = (
            "confidence_state_created"
            if created
            else "confidence_state_updated"
        )
        expert["lifecycle_state"] = "active"
        expert["last_channel_support"] = channel_support
        expert["scoring_weight"] = float(
            profile["reference_confidence"]
        )
        expert["state_confidence"] = float(
            profile["state_confidence"]
        )
        expert["normality_confidence"] = float(
            profile["normality_confidence"]
        )
        expert["reference_confidence"] = float(
            profile["reference_confidence"]
        )
        expert["state_effective_support"] = float(
            profile["effective_support"]
        )
        expert["state_member_image_ids"] = member_ids
        expert["state_member_weights"] = [
            float(weight) for weight in weights
        ]
        expert["lab_state"] = self._build_lab_state(
            self.bootstrap_confidence_members,
            existing_state=expert.get("lab_state"),
            freeze_anchor=bool(profile["reference_confidence"] > 0.0),
            step=step,
        )
        if (
            self.candidate_reliability_enabled
            and self.candidate_patch_reliability_mode
            == "cross_patch_robust"
        ):
            expert["member_reliability"] = float(
                representative.get("member_reliability", 1.0)
            )
            expert["scoring_patch_reliabilities"] = [
                float(value)
                for value in torch.as_tensor(
                    representative.get(
                        "scoring_patch_reliabilities",
                        representative["patch_reliabilities"],
                    ),
                    dtype=torch.float32,
                ).tolist()
            ]
            expert["reference_reliability_penalty"] = float(
                self.reference_reliability_penalty
            )
            expert["reliability_audit"] = dict(
                representative.get("reliability_audit", {})
            )
            expert["reference_member_image_ids"] = list(member_ids)
            expert["reference_member_reliabilities"] = [
                float(item.get("member_reliability", 1.0))
                for item in self.bootstrap_confidence_members
            ]
        self._register_expert_historical_state(expert, "active", step)

        self.last_bootstrap_event = {
            "decision": expert["last_event"],
            "pool_size": len(self.bootstrap_confidence_members),
            "selected_image_ids": member_ids,
            "representative_image_id": int(representative["image_id"]),
            "state_confidence": float(profile["state_confidence"]),
            "normality_confidence": float(
                profile["normality_confidence"]
            ),
            "reference_confidence": float(
                profile["reference_confidence"]
            ),
            "prequential_support": (
                prequential_profile["combined_support"]
                if prequential_profile is not None
                else None
            ),
            "prequential_profile": prequential_profile,
            "prequential_support_count": int(
                profile["prequential_support_count"]
            ),
            "state_effective_support": float(profile["effective_support"]),
            "state_coherence": float(profile["coherence"]),
            "member_weights": [float(weight) for weight in weights],
            "forced": False,
        }
        return (
            True,
            cluster_id,
            expert["expert_id"] if created else None,
            "bootstrap_confidence",
        )

    def _advance_cold_start(
        self,
        step,
        image_id,
        dino_patch_features,
        ms_score,
        channel_support,
        provisional_channel_support,
        lab_feature,
        patch_reliabilities=None,
    ):
        embedding = self._image_embedding(dino_patch_features)
        member = {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": embedding,
            "patch_prototypes": self._patch_coreset(
                dino_patch_features
            ),
            "quality": float(channel_support or 0.0) - float(ms_score or 0.0),
            "ms_score": ms_score,
            "channel_support": channel_support,
            "ms_threshold": None,
            "support_threshold": None,
            "candidate_route": "bootstrap",
            "provisional_channel_support": provisional_channel_support,
            "lab_feature": self._lab_feature(lab_feature),
        }
        self._attach_candidate_reliability(
            member,
            dino_patch_features,
            patch_reliabilities,
        )

        if self.bootstrap_provisional is not None:
            provisional = self.bootstrap_provisional
            similarity = float(embedding @ provisional["embedding_center"])
            support_values = [
                float(value)
                for value in (provisional_channel_support, channel_support)
                if value is not None
            ]
            observed_support = max(support_values) if support_values else None
            support_ok = bool(
                observed_support is not None
                and observed_support
                >= self.cold_start_provisional_min_channel_support
            )
            risk_ceiling = provisional.get("risk_ceiling")
            risk_ok = bool(
                ms_score is not None
                and np.isfinite(float(ms_score))
                and (
                    risk_ceiling is None
                    or float(ms_score) <= float(risk_ceiling)
                )
            )
            consistent = bool(
                similarity >= self.duplicate_similarity
                and support_ok
                and risk_ok
            )
            provisional["wait_steps"] += 1
            provisional["confirm_streak"] = (
                provisional["confirm_streak"] + 1 if consistent else 0
            )
            provisional["last_similarity"] = similarity
            provisional["last_channel_support"] = observed_support
            provisional["last_ms_score"] = ms_score
            if (
                provisional["confirm_streak"]
                >= self.cold_start_provisional_confirm_steps
            ):
                return self._admit_bootstrap_expert(
                    provisional["members"],
                    provisional["similarities"],
                    step,
                    provisional["pool_size"],
                    provisional["minimum_similarity"],
                    forced=False,
                    confirmation_steps=provisional["confirm_streak"],
                )
            if (
                provisional["wait_steps"]
                >= self.cold_start_provisional_max_wait_steps
            ):
                provisional_ids = [
                    int(item["image_id"]) for item in provisional["members"]
                ]
                self.bootstrap_provisional = None
                self.bootstrap_members = [member]
                self.last_bootstrap_event = {
                    "decision": "rolled_back",
                    "pool_size": int(provisional["pool_size"]),
                    "selected_image_ids": [],
                    "provisional_image_ids": provisional_ids,
                    "minimum_similarity": float(
                        provisional["minimum_similarity"]
                    ),
                    "confirmation_similarity": similarity,
                    "confirmation_channel_support": observed_support,
                    "confirmation_ms_score": ms_score,
                    "confirmation_risk_ceiling": risk_ceiling,
                    "confirmation_steps": int(provisional["confirm_streak"]),
                    "forced": False,
                }
                return True, None, None, "bootstrap"
            self.last_bootstrap_event = {
                "decision": "provisional_confirming",
                "pool_size": int(provisional["pool_size"]),
                "selected_image_ids": [],
                "provisional_image_ids": [
                    int(item["image_id"]) for item in provisional["members"]
                ],
                "minimum_similarity": float(
                    provisional["minimum_similarity"]
                ),
                "confirmation_similarity": similarity,
                "confirmation_channel_support": observed_support,
                "confirmation_ms_score": ms_score,
                "confirmation_risk_ceiling": risk_ceiling,
                "confirmation_steps": int(provisional["confirm_streak"]),
                "forced": False,
            }
            return True, None, None, "bootstrap"

        self.bootstrap_members.append(member)
        if (
            self.cold_start_provisional_enabled
            and len(self.bootstrap_members) > self.cold_start_max_samples
        ):
            self.bootstrap_members = self.bootstrap_members[
                -self.cold_start_max_samples:
            ]
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
            mean_quality = float(
                np.mean([item["quality"] for item in selected])
            )
            score = (
                minimum,
                mean,
                mean_quality,
                tuple(-index for index in indices),
            )
            if best is None or score > best[0]:
                best = (score, indices, selected, similarities)

        minimum_similarity = best[0][0]
        forced = bool(
            not self.cold_start_provisional_enabled
            and pool_size >= self.cold_start_max_samples
        )
        if minimum_similarity < self.duplicate_similarity and not forced:
            return True, None, None, "bootstrap"

        _, _, selected, similarities = best
        if self.cold_start_provisional_enabled:
            selected_risks = [
                float(item["ms_score"])
                for item in selected
                if item.get("ms_score") is not None
                and np.isfinite(float(item["ms_score"]))
            ]
            embedding_center = F.normalize(
                torch.stack([item["embedding"] for item in selected]).sum(dim=0),
                dim=0,
            )
            self.bootstrap_provisional = {
                "members": selected,
                "similarities": similarities,
                "embedding_center": embedding_center,
                "start_step": int(step),
                "pool_size": int(pool_size),
                "minimum_similarity": float(minimum_similarity),
                "wait_steps": 0,
                "confirm_streak": 0,
                "last_similarity": None,
                "last_channel_support": None,
                "last_ms_score": None,
                "risk_ceiling": (
                    max(selected_risks) if selected_risks else None
                ),
            }
            self.last_bootstrap_event = {
                "decision": "provisional_created",
                "pool_size": int(pool_size),
                "selected_image_ids": [],
                "provisional_image_ids": [
                    int(item["image_id"]) for item in selected
                ],
                "minimum_similarity": float(minimum_similarity),
                "risk_ceiling": self.bootstrap_provisional["risk_ceiling"],
                "confirmation_steps": 0,
                "forced": False,
            }
            return True, None, None, "bootstrap"
        return self._admit_bootstrap_expert(
            selected,
            similarities,
            step,
            pool_size,
            minimum_similarity,
            forced=forced,
        )

    def advance_lab_state(self, lab_feature, step=None, image_id=None):
        feature = self._lab_feature(lab_feature)
        if feature is None:
            return None
        candidates = [
            expert
            for expert in self.active_experts
            if expert.get("lab_state") is not None
            and expert["lab_state"].get("detector_ready", True)
            and (
                not self.photometric_stable_lab_anchor_enabled
                or expert["lab_state"].get("anchor_frozen", False)
            )
        ]
        if not candidates:
            return None

        profiles = []
        for expert in candidates:
            state = expert["lab_state"]
            reference_center = np.asarray(
                state.get(
                    "anchor_center",
                    state.get("detector_center", state["center"]),
                ),
                dtype=np.float64,
            )
            distances = self._lab_distances(
                feature,
                reference_center,
            )
            profiles.append((distances["total_distance"], expert, distances))
        _, expert, distances = min(profiles, key=lambda item: item[0])
        state = expert["lab_state"]
        record_step = self.last_step + 1 if step is None else int(step)
        record_image_id = record_step if image_id is None else int(image_id)
        current_record = {
            "step": record_step,
            "image_id": record_image_id,
            "feature": feature.tolist(),
        }
        relevant_profile = None
        if self.photometric_relevant_history_enabled:
            relevant_profile = self._relevant_lab_history_profile(
                state,
                current_record,
            )
            history = list(state.get("recent_lab_history", []))
            history.append(current_record)
            state["recent_lab_history"] = history[-self.photometric_history_max:]
        channel_hits = {}
        for channel in ("l", "ab"):
            state[f"{channel}_ewma"] = float(
                state["alpha"] * distances[f"{channel}_distance"]
                + (1.0 - state["alpha"]) * state[f"{channel}_ewma"]
            )
            hit = state[f"{channel}_ewma"] > state[f"{channel}_threshold"]
            channel_hits[channel] = bool(hit)
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
        candidate_center = None
        candidate_distances = None
        candidate_history = []
        if self.photometric_stable_lab_anchor_enabled:
            if any(channel_hits.values()):
                candidate_history = list(
                    state.get("candidate_lab_history", [])
                )
                candidate_history.append(current_record)
                candidate_history = candidate_history[
                    -self.photometric_history_max:
                ]
            state["candidate_lab_history"] = candidate_history
            if candidate_history:
                candidate_center = np.median(
                    np.stack(
                        [
                            np.asarray(record["feature"], dtype=np.float64)
                            for record in candidate_history
                        ],
                        axis=0,
                    ),
                    axis=0,
                )
                candidate_distances = self._lab_distances(
                    candidate_center,
                    np.asarray(state["anchor_center"], dtype=np.float64),
                )
        anchor_center = np.asarray(
            state.get(
                "anchor_center",
                state.get("detector_center", state["center"]),
            ),
            dtype=np.float64,
        )
        coverage_center = np.asarray(
            state.get("coverage_center", state["center"]),
            dtype=np.float64,
        )
        anchor_to_coverage = self._lab_distances(
            coverage_center,
            anchor_center,
        )
        return {
            "expert_id": int(expert["expert_id"]),
            "current_lab_feature": feature.copy().tolist(),
            "lab_center": anchor_center.copy().tolist(),
            "stable_lab_anchor_enabled": bool(
                self.photometric_stable_lab_anchor_enabled
            ),
            "anchor_frozen": bool(state.get("anchor_frozen", False)),
            "lab_anchor_center": anchor_center.copy().tolist(),
            "lab_coverage_center": coverage_center.copy().tolist(),
            "candidate_lab_center": (
                candidate_center.copy().tolist()
                if candidate_center is not None
                else None
            ),
            "candidate_lab_count": len(candidate_history),
            "anchor_to_coverage_l_distance": anchor_to_coverage[
                "l_distance"
            ],
            "anchor_to_coverage_ab_distance": anchor_to_coverage[
                "ab_distance"
            ],
            "anchor_to_coverage_total_distance": anchor_to_coverage[
                "total_distance"
            ],
            "anchor_to_candidate_l_distance": (
                candidate_distances["l_distance"]
                if candidate_distances is not None
                else None
            ),
            "anchor_to_candidate_ab_distance": (
                candidate_distances["ab_distance"]
                if candidate_distances is not None
                else None
            ),
            "anchor_to_candidate_total_distance": (
                candidate_distances["total_distance"]
                if candidate_distances is not None
                else None
            ),
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
        handled_distance = None
        handled_state_blocked = False
        if self.photometric_handled_lab_center is not None:
            current_lab_feature = lab_drift_profile.get("current_lab_feature")
            if current_lab_feature is not None:
                handled_distance = self._lab_distances(
                    np.asarray(current_lab_feature, dtype=np.float64),
                    self.photometric_handled_lab_center,
                )["total_distance"]
                if (
                    handled_distance
                    > self.photometric_local_lab_distance_threshold
                ):
                    self.photometric_rearm_streak += 1
                else:
                    self.photometric_rearm_streak = 0
                if (
                    self.photometric_rearm_streak
                    >= self.photometric_exit_confirm_steps
                ):
                    self.photometric_handled_lab_center = None
                    self.photometric_rearm_streak = 0
                else:
                    handled_state_blocked = True
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
            and not handled_state_blocked
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
            "handled_state_blocked": handled_state_blocked,
            "handled_state_distance": handled_distance,
            "rearm_streak": int(self.photometric_rearm_streak),
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

    def _finish_photometric_episode(self, handled_lab_center=None):
        if (
            self.photometric_episode_enabled
            and self.last_novel_quarantine_event is not None
        ):
            self.last_novel_quarantine_event["episode_finished"] = True
        if self.photometric_episode_enabled and handled_lab_center is not None:
            self.photometric_handled_lab_center = np.asarray(
                handled_lab_center,
                dtype=np.float64,
            ).copy()
            self.photometric_rearm_streak = 0
        self._reset_photometric_episode()

    @staticmethod
    def _photometric_pool_lab_center(pool):
        features = [
            np.asarray(member["lab_feature"], dtype=np.float64)
            for member in pool
            if member.get("lab_feature") is not None
        ]
        if not features:
            return None
        return np.median(np.stack(features, axis=0), axis=0)

    def _abort_photometric_quarantine(self, step, reason):
        pool = list(self.novel_quarantine or [])
        handled_lab_center = self._photometric_pool_lab_center(
            [
                member
                for member in pool
                if member.get("historical_low_risk", False)
            ]
            or pool
        )
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
        self._finish_photometric_episode(handled_lab_center)

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
        required_count=None,
    ):
        required_count = int(
            self.photometric_min_low_risk_members
            if required_count is None
            else required_count
        )
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
            embeddings = torch.stack(
                [member["embedding"] for member in pool],
                dim=0,
            )
            similarities = embeddings @ embeddings.T
            joint_neighbors = (
                pairwise_distances
                <= self.photometric_local_lab_distance_threshold
            ) & (
                similarities.detach().cpu().numpy()
                >= self.photometric_dino_similarity_floor
            )
            joint_neighbor_counts = joint_neighbors.sum(axis=1)
            center_index = int(joint_neighbor_counts.argmax())
            selected_indices = np.flatnonzero(joint_neighbors[center_index])
            lab_inlier_count = int(
                np.sum(
                    pairwise_distances[center_index]
                    <= self.photometric_local_lab_distance_threshold
                )
            )
            dino_inlier_count = int(
                torch.sum(
                    similarities[center_index]
                    >= self.photometric_dino_similarity_floor
                ).item()
            )
            selected_normalized = normalized[selected_indices]
            selected_features = np.stack(
                [
                    np.asarray(features[index], dtype=np.float64)
                    for index in selected_indices
                ],
                axis=0,
            )
            local_center = np.median(selected_normalized, axis=0)
            selected_lab_center = np.median(selected_features, axis=0)
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
                and len(selected_indices)
                >= required_count
            )
            selected_image_ids = [
                int(pool[index]["image_id"])
                for index in selected_indices
            ]
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
                "required_count": required_count,
                "lab_inlier_count": lab_inlier_count,
                "dino_inlier_count": dino_inlier_count,
                "selected_inlier_count": len(selected_indices),
                "selected_inlier_image_ids": selected_image_ids,
                "selected_lab_center": selected_lab_center.tolist(),
                "local_lab_distance_threshold": (
                    self.photometric_local_lab_distance_threshold
                ),
                "dino_similarity_floor": (
                    self.photometric_dino_similarity_floor
                ),
                "local_distance_max": float(
                    pairwise_distances[center_index, selected_indices].max()
                ),
                "local_distance_median": float(
                    np.median(
                        pairwise_distances[center_index, selected_indices]
                    )
                ),
                "l_separation": l_separation,
                "ab_separation": ab_separation,
                "selected_joint_center_image_id": int(
                    pool[center_index]["image_id"]
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

    def _advance_dormant_lifecycle(
        self,
        step,
        support_threshold,
        expert_channel_supports,
        current_expert_supports,
    ):
        lifecycle_events = []
        for expert in self.active_experts:
            expert_id = int(expert["expert_id"])
            state_before = expert.get("lifecycle_state", "active")
            ttl_before = int(expert["ttl"])
            patience_before = int(expert["patience"])
            last_supported_step_before = int(
                expert["last_supported_step"]
            )
            memory_support = self._optional_score(
                f"expert_channel_supports[{expert_id}]",
                expert_channel_supports.get(expert_id),
                unit_interval=True,
            )
            current_support = self._optional_score(
                f"current_expert_supports[{expert_id}]",
                current_expert_supports.get(expert_id),
                unit_interval=True,
            )
            support_gap = None
            became_dormant = False
            reactivated = False

            if expert.get("admission_route") == "bootstrap_confidence":
                expert["lifecycle_state"] = "active"
                expert["last_event"] = "confidence_managed"
            elif state_before == "dormant":
                if (
                    current_support is not None
                    and support_threshold is not None
                    and current_support >= support_threshold
                ):
                    expert["lifecycle_state"] = "active"
                    expert["ttl"] = expert["patience"]
                    expert["last_supported_step"] = int(step)
                    expert["support_count"] += 1
                    expert["last_event"] = "reactivated"
                    reactivated = True
                else:
                    expert["lifecycle_state"] = "dormant"
                    expert["ttl"] = 0
                    expert["last_event"] = "dormant"
            elif memory_support is None or support_threshold is None:
                expert["lifecycle_state"] = "active"
                expert["last_event"] = "signal_unavailable"
            elif memory_support >= support_threshold:
                support_gap = step - expert["last_supported_step"]
                expert["max_support_gap"] = max(
                    expert["max_support_gap"], support_gap
                )
                expert["patience"] = self._patience_from_gap(
                    expert["max_support_gap"]
                )
                expert["ttl"] = expert["patience"]
                expert["last_supported_step"] = int(step)
                expert["support_count"] += 1
                expert["lifecycle_state"] = "active"
                expert["last_event"] = "ttl_refreshed"
            else:
                expert["ttl"] -= 1
                if expert["ttl"] <= 0:
                    expert["ttl"] = 0
                    expert["lifecycle_state"] = "dormant"
                    expert["last_event"] = "became_dormant"
                    became_dormant = True
                else:
                    expert["lifecycle_state"] = "active"
                    expert["last_event"] = "ttl_decrement"

            expert["last_channel_support"] = memory_support
            state_after = expert.get("lifecycle_state", "active")
            lifecycle_events.append(
                {
                    "step": int(step),
                    "expert_id": expert_id,
                    "expert_image_id": int(expert["image_id"]),
                    "cluster_id": int(expert["cluster_id"]),
                    "admission_step": int(expert["admission_step"]),
                    "age": int(step - expert["admission_step"]),
                    "channel_support": memory_support,
                    "current_expert_support": current_support,
                    "support_threshold": support_threshold,
                    "support_available": memory_support is not None,
                    "threshold_available": support_threshold is not None,
                    "support_gap": support_gap,
                    "ttl_before": ttl_before,
                    "ttl_after": int(expert["ttl"]),
                    "patience_before": patience_before,
                    "patience_after": int(expert["patience"]),
                    "last_supported_step_before": last_supported_step_before,
                    "last_supported_step_after": int(
                        expert["last_supported_step"]
                    ),
                    "signal_decision": expert["last_event"],
                    "decision": expert["last_event"],
                    "refreshed": expert["last_event"] == "ttl_refreshed",
                    "deleted_this_step": False,
                    "alive_after": True,
                    "active_after": state_after == "active",
                    "lifecycle_state_before": state_before,
                    "lifecycle_state_after": state_after,
                    "became_dormant_this_step": became_dormant,
                    "reactivated_this_step": reactivated,
                    "coverage_floor_retained": False,
                    "minimum_active_experts": self.minimum_active_experts,
                    "admission_candidate_image_id": None,
                    "admission_candidate_ms_score": None,
                    "admission_ms_threshold": None,
                    "admission_candidate_channel_support": None,
                    "admission_support_threshold": None,
                    "representative_mode": expert["representative_mode"],
                    "admission_ttl_mode": expert["admission_ttl_mode"],
                    "admission_route": expert.get(
                        "admission_route", "legacy"
                    ),
                }
            )
        return [], lifecycle_events

    #管理存活专家
    def _advance_lifecycle(
        self,
        step,
        support_threshold,
        expert_channel_supports,
        current_expert_supports=None,
    ):
        if self.lifecycle_mode == "dormant_replace":
            return self._advance_dormant_lifecycle(
                step,
                support_threshold,
                expert_channel_supports,
                current_expert_supports or {},
            )
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

            if expert.get("admission_route") == "bootstrap_confidence":
                event = "confidence_managed"
            elif (
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
                    "tentative_delete": (
                        expert["ttl"] <= 0
                        and expert.get("admission_route")
                        != "bootstrap_confidence"
                    ),
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
                self._register_expert_historical_state(
                    expert,
                    "retired",
                    step,
                )
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
        current_expert_supports=None,
        patch_reliabilities=None,
    ):
        member = {
            "step": int(step),
            "image_id": int(image_id),
            "embedding": self._image_embedding(dino_patch_features),
            "patch_prototypes": self._patch_coreset(
                dino_patch_features,
                patch_ids=(
                    None
                    if self.candidate_reliability_enabled
                    else memory_patch_ids
                ),
                patch_reliabilities=(
                    patch_reliabilities
                    if self.candidate_reliability_enabled
                    else None
                ),
            ),
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
            "current_expert_supports": self._normalized_expert_supports(
                current_expert_supports
            ),
        }
        if self.candidate_reliability_enabled:
            if patch_reliabilities is None:
                patch_reliabilities = torch.ones(
                    dino_patch_features.shape[0],
                    dtype=torch.float32,
                )
            base_reliabilities = torch.as_tensor(
                patch_reliabilities,
                dtype=torch.float32,
            ).detach().cpu()
            member["base_patch_reliabilities"] = base_reliabilities
            member["patch_reliabilities"] = base_reliabilities.clone()
            member["reliability_patch_bank"] = self._candidate_patch_bank(
                dino_patch_features,
                base_reliabilities,
            )
            member["member_reliability"] = 0.5
        return member

    def _select_irbank_members(self, pool, min_keep=1):
        if self.candidate_reliability_enabled:
            eligible = list(pool)
            self._refresh_candidate_reliabilities(eligible)
            return sorted(
                eligible,
                key=lambda item: (
                    float(item["ms_short_score"])
                    / max(
                        float(item.get("member_reliability", 1.0)),
                        self.candidate_reliability_floor,
                    ),
                    int(item["step"]),
                ),
            )
        eligible = [
            member for member in pool if member.get("memory_patch_ids")
        ]
        if not eligible:
            return []
        keep = min(
            len(eligible),
            max(
                int(min_keep),
                int(
                    math.ceil(
                        len(eligible) * self.novel_memory_keep_fraction
                    )
                ),
            ),
        )
        return sorted(
            eligible,
            key=lambda item: item["ms_short_score"],
        )[:keep]

    def _fitness_history_profile(self, history, current_member):
        current_step = int(current_member["step"])
        current_image_id = int(current_member["image_id"])
        history = [
            member
            for member in list(history)[-self.photometric_history_max:]
            if int(member["step"]) < current_step
            and int(member["image_id"]) != current_image_id
        ]
        if not history:
            return [], {
                "available_count": 0,
                "selected_count": 0,
                "effective_history_size": 0.0,
                "selected_image_ids": [],
                "member_profiles": [],
            }

        ages = np.asarray(
            [max(current_step - int(member["step"]), 1) for member in history],
            dtype=np.float64,
        )
        recency_scale = max(float(np.median(ages)), 1.0)
        current_embedding = current_member["embedding"]
        current_lab = current_member.get("lab_feature")
        current_supports = current_member.get("current_expert_supports", {})
        profiles = []

        for member, age in zip(history, ages):
            lab_similarity = 1.0
            member_lab = member.get("lab_feature")
            if current_lab is not None and member_lab is not None:
                lab_distance = self._lab_distances(
                    np.asarray(current_lab, dtype=np.float64),
                    np.asarray(member_lab, dtype=np.float64),
                )["total_distance"]
                lab_similarity = math.exp(-lab_distance)
            else:
                lab_distance = None

            dino_similarity = float(
                F.cosine_similarity(
                    current_embedding.reshape(1, -1),
                    member["embedding"].reshape(1, -1),
                ).item()
            )
            dino_affinity = min(max((dino_similarity + 1.0) / 2.0, 0.0), 1.0)

            member_supports = member.get("current_expert_supports", {})
            shared_expert_ids = sorted(
                set(current_supports).intersection(member_supports)
            )
            if shared_expert_ids:
                channel_affinity = float(
                    np.mean(
                        [
                            1.0
                            - abs(
                                float(current_supports[expert_id])
                                - float(member_supports[expert_id])
                            )
                            for expert_id in shared_expert_ids
                        ]
                    )
                )
            elif (
                current_member.get("channel_support") is not None
                and member.get("channel_support") is not None
            ):
                channel_affinity = 1.0 - abs(
                    float(current_member["channel_support"])
                    - float(member["channel_support"])
                )
            else:
                channel_affinity = 1.0
            channel_affinity = min(max(channel_affinity, 0.0), 1.0)
            recency_affinity = math.exp(-float(age) / recency_scale)
            similarity_components = np.asarray(
                [
                    lab_similarity,
                    dino_affinity,
                    channel_affinity,
                ],
                dtype=np.float64,
            )
            state_similarity = float(
                np.exp(
                    np.mean(
                        np.log(np.maximum(similarity_components, 1e-12))
                    )
                )
            )
            relevance = float(state_similarity * recency_affinity)
            profiles.append(
                {
                    "member": member,
                    "image_id": int(member["image_id"]),
                    "step": int(member["step"]),
                    "age": int(age),
                    "lab_distance": lab_distance,
                    "lab_similarity": float(lab_similarity),
                    "dino_similarity": dino_similarity,
                    "dino_affinity": dino_affinity,
                    "channel_affinity": channel_affinity,
                    "recency_affinity": recency_affinity,
                    "state_similarity": state_similarity,
                    "relevance": relevance,
                }
            )

        relevances = np.asarray(
            [profile["relevance"] for profile in profiles], dtype=np.float64
        )
        effective_size = float(relevances.sum())
        keep = min(len(profiles), max(1, int(math.ceil(effective_size))))
        selected_profiles = sorted(
            profiles,
            key=lambda profile: (
                -profile["relevance"],
                -profile["step"],
                profile["image_id"],
            ),
        )[:keep]
        selected_members = []
        for profile in selected_profiles:
            member = dict(profile["member"])
            member["history_relevance"] = float(profile["relevance"])
            selected_members.append(member)

        return selected_members, {
            "available_count": len(profiles),
            "selected_count": len(selected_members),
            "effective_history_size": effective_size,
            "recency_scale": recency_scale,
            "selected_image_ids": [
                int(profile["image_id"]) for profile in selected_profiles
            ],
            "relevance_min": float(relevances.min()),
            "relevance_max": float(relevances.max()),
            "relevance_mean": float(relevances.mean()),
            "member_profiles": [
                {
                    key: value
                    for key, value in profile.items()
                    if key != "member"
                }
                for profile in profiles
            ],
        }

    def _normalized_expert_supports(self, supports):
        normalized = {}
        for expert_id, support in (supports or {}).items():
            parsed = self._optional_score(
                f"current_expert_supports[{expert_id}]",
                support,
                unit_interval=True,
            )
            if parsed is not None:
                normalized[int(expert_id)] = float(parsed)
        return normalized

    def _record_coverage_observation(
        self,
        step,
        ms_score,
        ms_threshold,
        support_threshold,
        current_expert_supports,
        dino_patch_features=None,
        memory_patch_ids=None,
        lab_feature=None,
    ):
        if (
            ms_score is None
            or ms_threshold is None
            or ms_score > ms_threshold
            or support_threshold is None
        ):
            return
        supports = self._normalized_expert_supports(
            current_expert_supports
        )
        if not supports:
            return
        self.coverage_history.append(
            {
                "step": int(step),
                "image_id": int(step),
                "support_threshold": float(support_threshold),
                "supports": supports,
                "embedding": (
                    self._image_embedding(dino_patch_features)
                    if dino_patch_features is not None
                    else None
                ),
                "patch_prototypes": (
                    self._patch_coreset(
                        dino_patch_features,
                        patch_ids=memory_patch_ids,
                    )
                    if dino_patch_features is not None
                    else None
                ),
                "lab_feature": self._lab_feature(lab_feature),
            }
        )
        del self.coverage_history[:-self.budget_coverage_window]

    def _candidate_new_coverage_fraction(
        self,
        candidate_members,
        support_threshold,
    ):
        if support_threshold is None:
            return None, 0, 0
        active_ids = {
            int(expert["expert_id"])
            for expert in self.active_experts
        }
        evaluated = 0
        uncovered = 0
        for member in candidate_members:
            supports = {
                int(expert_id): float(support)
                for expert_id, support in member.get(
                    "current_expert_supports", {}
                ).items()
                if int(expert_id) in active_ids
            }
            if not supports:
                continue
            evaluated += 1
            if not any(
                support >= support_threshold
                for support in supports.values()
            ):
                uncovered += 1
        if evaluated == 0:
            return None, 0, 0
        return float(uncovered / evaluated), uncovered, evaluated

    def _historical_state_dino_center(self, cluster_id):
        if cluster_id is None or not 0 <= int(cluster_id) < len(self.clusters):
            return None
        embedding_sum = self.clusters[int(cluster_id)].get("embedding_sum")
        if embedding_sum is None or float(embedding_sum.norm()) == 0.0:
            return None
        return F.normalize(embedding_sum, dim=0).detach().cpu()

    def _historical_member_records(self, members):
        records = []
        for member in members:
            lab_feature = member.get("lab_feature")
            embedding = member.get("embedding")
            if lab_feature is None or embedding is None:
                continue
            records.append(
                {
                    "step": int(member.get("step", self.last_step)),
                    "image_id": int(member["image_id"]),
                    "lab_feature": np.asarray(
                        lab_feature, dtype=np.float64
                    ).copy(),
                    "embedding": F.normalize(
                        embedding.detach().cpu(), dim=0
                    ),
                    "memory_patch_ids": list(
                        member.get("memory_patch_ids", [])
                    ),
                    "member_reliability": float(
                        member.get("member_reliability", 1.0)
                    ),
                }
            )
        return records[-self.photometric_history_max:]

    def _historical_state_distribution(self, member_records):
        if not member_records:
            return None
        lab_features = np.stack(
            [record["lab_feature"] for record in member_records], axis=0
        )
        reliability_weights = np.asarray(
            [
                float(record.get("member_reliability", 1.0))
                for record in member_records
            ],
            dtype=np.float64,
        )
        lab_center = (
            np.average(
                lab_features,
                axis=0,
                weights=reliability_weights,
            )
            if self.candidate_reliability_enabled
            else np.median(lab_features, axis=0)
        )
        profiles = [
            self._lab_distances(feature, lab_center)
            for feature in lab_features
        ]

        radii = {}
        for channel, floor in (
            ("l", self.photometric_l_ewma_floor),
            ("ab", self.photometric_ab_ewma_floor),
        ):
            distances = np.asarray(
                [profile[f"{channel}_distance"] for profile in profiles],
                dtype=np.float64,
            )
            median = float(np.median(distances))
            mad = float(np.median(np.abs(distances - median)))
            radii[channel] = max(
                median + self.lab_ewma_threshold_k * 1.4826 * mad,
                float(floor),
            )

        dino_stack = torch.stack(
            [record["embedding"] for record in member_records]
        )
        if self.candidate_reliability_enabled:
            dino_stack = dino_stack * torch.as_tensor(
                reliability_weights,
                dtype=dino_stack.dtype,
            )[:, None]
        dino_center = F.normalize(
            dino_stack.sum(dim=0),
            dim=0,
        ).detach().cpu()
        return {
            "lab_center": lab_center,
            "l_radius": float(radii["l"]),
            "ab_radius": float(radii["ab"]),
            "dino_center": dino_center,
            "member_image_ids": [
                int(record["image_id"]) for record in member_records
            ],
            "member_reliabilities": reliability_weights.tolist(),
        }

    def _record_state_lifecycle_event(
        self,
        event_type,
        step,
        state,
        *,
        matched_or_merged_state_id=None,
        source_type_before=None,
        status_before=None,
        reason=None,
    ):
        event = {
            "event_type": str(event_type),
            "step": int(step),
            "state_id": int(state["state_id"]),
            "matched_or_merged_state_id": (
                int(matched_or_merged_state_id)
                if matched_or_merged_state_id is not None
                else None
            ),
            "source_type_before": source_type_before,
            "source_type_after": state.get("source_type"),
            "status_before": status_before,
            "status_after": state.get("state_status", "stable"),
            "active_expert_id": state.get("active_expert_id"),
            "support_count": int(state.get("support_count", 0)),
            "support_event_count": int(
                state.get("support_event_count", 0)
            ),
            "member_count": len(state.get("member_records", [])),
            "temporary_effective_support": (
                float(state["temporary_effective_support"])
                if state.get("temporary_effective_support") is not None
                else None
            ),
            "temporary_support_increment": (
                float(state["temporary_support_increment"])
                if state.get("temporary_support_increment") is not None
                else None
            ),
            "temporary_dino_evidence": (
                float(state["temporary_dino_evidence"])
                if state.get("temporary_dino_evidence") is not None
                else None
            ),
            "temporary_channel_evidence": (
                float(state["temporary_channel_evidence"])
                if state.get("temporary_channel_evidence") is not None
                else None
            ),
            "temporary_risk_evidence": (
                float(state["temporary_risk_evidence"])
                if state.get("temporary_risk_evidence") is not None
                else None
            ),
            "reason": reason,
        }
        self.state_lifecycle_events.append(event)
        return event

    def _historical_states_match(self, left, right):
        left_center = left.get("lab_center")
        right_center = right.get("lab_center")
        left_embedding = left.get("dino_center")
        right_embedding = right.get("dino_center")
        if (
            left_center is None
            or right_center is None
            or left_embedding is None
            or right_embedding is None
        ):
            return None
        distances = self._lab_distances(
            np.asarray(left_center, dtype=np.float64),
            np.asarray(right_center, dtype=np.float64),
        )
        l_threshold = max(
            float(left.get("l_radius") or left.get("l_threshold") or 0.0),
            self.photometric_l_ewma_floor,
        ) + max(
            float(right.get("l_radius") or right.get("l_threshold") or 0.0),
            self.photometric_l_ewma_floor,
        )
        ab_threshold = max(
            float(left.get("ab_radius") or left.get("ab_threshold") or 0.0),
            self.photometric_ab_ewma_floor,
        ) + max(
            float(right.get("ab_radius") or right.get("ab_threshold") or 0.0),
            self.photometric_ab_ewma_floor,
        )
        similarity = float(left_embedding @ right_embedding)
        l_relative = distances["l_distance"] / max(l_threshold, 1e-12)
        ab_relative = distances["ab_distance"] / max(ab_threshold, 1e-12)
        dino_strength = max(
            0.0,
            min(
                1.0,
                (
                    similarity - self.photometric_dino_similarity_floor
                )
                / max(1.0 - self.photometric_dino_similarity_floor, 1e-12),
            ),
        )
        lab_strength = max(0.0, 1.0 - max(l_relative, ab_relative))
        return {
            "matched": bool(
                l_relative <= 1.0
                and ab_relative <= 1.0
                and similarity >= self.photometric_dino_similarity_floor
            ),
            "joint_match_strength": float(
                (lab_strength + dino_strength) / 2.0
            ),
            "l_relative_distance": float(l_relative),
            "ab_relative_distance": float(ab_relative),
            "dino_similarity": similarity,
        }

    def _merge_historical_state_pair(self, keep, remove, step):
        keep_records = list(keep.get("member_records", []))
        remove_records = list(remove.get("member_records", []))
        unique_records = {}
        for record in keep_records + remove_records:
            unique_records[(int(record["step"]), int(record["image_id"]))] = record
        merged_records = list(unique_records.values())[
            -self.photometric_history_max:
        ]
        distribution = self._historical_state_distribution(merged_records)
        source_before = keep.get("source_type")
        status_before = keep.get("state_status", "stable")
        if distribution is not None:
            keep["lab_center"] = distribution["lab_center"]
            keep["dino_center"] = distribution["dino_center"]
            keep["l_radius"] = distribution["l_radius"]
            keep["ab_radius"] = distribution["ab_radius"]
            keep["l_threshold"] = distribution["l_radius"]
            keep["ab_threshold"] = distribution["ab_radius"]
            keep["member_image_ids"] = distribution["member_image_ids"]
        keep["member_records"] = merged_records
        keep["support_count"] = int(keep.get("support_count", 0)) + int(
            remove.get("support_count", 0)
        )
        keep["support_event_count"] = int(
            keep.get("support_event_count", 0)
        ) + int(remove.get("support_event_count", 0))
        if self.photometric_denstream_support_enabled:
            keep["temporary_effective_support"] = float(
                keep.get("temporary_effective_support", 0.0)
                + remove.get("temporary_effective_support", 0.0)
            )
            keep["temporary_support_last_step"] = max(
                int(keep.get("temporary_support_last_step", -1)),
                int(remove.get("temporary_support_last_step", -1)),
            )
        keep["last_seen_step"] = max(
            int(keep.get("last_seen_step", -1)),
            int(remove.get("last_seen_step", -1)),
        )
        if (
            self.photometric_denstream_support_enabled
            and keep.get("state_status", "stable") == "temporary"
            and keep.get("temporary_effective_support", 0.0)
            >= self.photometric_denstream_mature_support
        ):
            keep["state_status"] = "stable"
            keep["source_type"] = "photometric_only"
        keep["has_patch_memory"] = bool(
            keep.get("has_patch_memory", False)
            or remove.get("has_patch_memory", False)
        )
        if (
            keep.get("state_status") != "stable"
            and remove.get("state_status") == "stable"
        ):
            keep["state_status"] = "stable"
            if keep.get("source_type") == "temporary":
                keep["source_type"] = remove.get(
                    "source_type", "photometric_only"
                )
        self.historical_states = [
            state
            for state in self.historical_states
            if int(state["state_id"]) != int(remove["state_id"])
        ]
        self._record_state_lifecycle_event(
            "merge",
            step,
            keep,
            matched_or_merged_state_id=remove["state_id"],
            source_type_before=source_before,
            status_before=status_before,
            reason="redundant_state_distribution",
        )
        if (
            status_before == "temporary"
            and keep.get("state_status") == "stable"
        ):
            self._record_state_lifecycle_event(
                "mature",
                step,
                keep,
                source_type_before=source_before,
                status_before=status_before,
                reason="denstream_merged_effective_support_matured",
            )

    def _merge_redundant_historical_states(self, step):
        if not self.photometric_state_distribution_enabled:
            return
        while True:
            candidates = []
            for index, left in enumerate(self.historical_states):
                for right in self.historical_states[index + 1:]:
                    left_active = left.get("active_expert_id")
                    right_active = right.get("active_expert_id")
                    if (
                        left_active is not None
                        and right_active is not None
                        and int(left_active) != int(right_active)
                    ):
                        continue
                    profile = self._historical_states_match(left, right)
                    if profile is None or not profile["matched"]:
                        continue
                    candidates.append(
                        (profile["joint_match_strength"], left, right)
                    )
            if not candidates:
                return
            _, left, right = max(candidates, key=lambda item: item[0])
            keep, remove = max(
                ((left, right), (right, left)),
                key=lambda pair: (
                    pair[0].get("active_expert_id") is not None,
                    bool(pair[0].get("has_patch_memory", False)),
                    pair[0].get("state_status", "stable") == "stable",
                    int(pair[0].get("support_event_count", 0)),
                    int(pair[0].get("support_count", 0)),
                    int(pair[0].get("last_seen_step", -1)),
                    -int(pair[0]["state_id"]),
                ),
            )
            self._merge_historical_state_pair(keep, remove, step)

    def _historical_state_redundancy(self, state):
        strengths = []
        for other in self.historical_states:
            if int(other["state_id"]) == int(state["state_id"]):
                continue
            profile = self._historical_states_match(state, other)
            if profile is not None:
                strengths.append(profile["joint_match_strength"])
        return max(strengths, default=0.0)

    def _trim_historical_states(self, step):
        self._merge_redundant_historical_states(step)
        while len(self.historical_states) > self.photometric_state_registry_cap:
            inactive = [
                state
                for state in self.historical_states
                if state.get("active_expert_id") is None
            ]
            if not inactive:
                raise RuntimeError(
                    "historical state registry is over capacity with only active states"
                )
            victim = min(
                inactive,
                key=lambda state: (
                    state.get("state_status", "stable") != "temporary",
                    -self._historical_state_redundancy(state),
                    int(state.get("support_event_count", 0)),
                    int(state.get("support_count", 0)),
                    int(state.get("last_seen_step", -1)),
                ),
            )
            self.historical_states.remove(victim)
            self._record_state_lifecycle_event(
                "evict",
                step,
                victim,
                source_type_before=victim.get("source_type"),
                status_before=victim.get("state_status", "stable"),
                reason="state_registry_capacity",
            )

    def _upsert_historical_state(
        self,
        *,
        source_type,
        step,
        lab_center,
        dino_center,
        support_count,
        source_expert_id=None,
        active_expert_id=None,
        cluster_id=None,
        l_threshold=None,
        ab_threshold=None,
        has_patch_memory=False,
        state_id=None,
        accumulate_support=False,
        member_records=None,
        state_status="stable",
        support_event_increment=0,
        temporary_support_evidence=None,
    ):
        if lab_center is None or dino_center is None:
            return None
        lab_center = np.asarray(lab_center, dtype=np.float64)
        dino_center = F.normalize(dino_center.detach().cpu(), dim=0)
        existing = None
        if state_id is not None:
            existing = next(
                (
                    state
                    for state in self.historical_states
                    if int(state["state_id"]) == int(state_id)
                ),
                None,
            )
        if existing is None and source_expert_id is not None:
            existing = next(
                (
                    state
                    for state in self.historical_states
                    if state.get("source_expert_id") is not None
                    and int(state["source_expert_id"])
                    == int(source_expert_id)
                ),
                None,
            )
        if (
            existing is None
            and source_type == "photometric_only"
            and not self.photometric_adaptive_candidate_evidence
        ):
            for state in self.historical_states:
                if state.get("source_type") != "photometric_only":
                    continue
                distances = self._lab_distances(
                    lab_center,
                    np.asarray(state["lab_center"], dtype=np.float64),
                )
                state_l_radius = max(
                    float(
                        state.get("l_radius")
                        or state.get("l_threshold")
                        or 0.0
                    ),
                    self.photometric_l_ewma_floor,
                )
                state_ab_radius = max(
                    float(
                        state.get("ab_radius")
                        or state.get("ab_threshold")
                        or 0.0
                    ),
                    self.photometric_ab_ewma_floor,
                )
                similarity = float(dino_center @ state["dino_center"])
                if (
                    distances["l_distance"] <= state_l_radius
                    and distances["ab_distance"] <= state_ab_radius
                    and similarity >= self.photometric_dino_similarity_floor
                ):
                    existing = state
                    break
        previous_records = list(existing.get("member_records", [])) if existing else []
        new_records = list(member_records or [])
        combined_records = previous_records + new_records
        unique_records = {}
        for record in combined_records:
            key = (int(record["step"]), int(record["image_id"]))
            unique_records[key] = record
        combined_records = list(unique_records.values())[
            -self.photometric_history_max:
        ]
        distribution = (
            self._historical_state_distribution(combined_records)
            if self.photometric_state_distribution_enabled
            else None
        )
        if distribution is not None:
            lab_center = distribution["lab_center"]
            dino_center = distribution["dino_center"]
            l_threshold = distribution["l_radius"]
            ab_threshold = distribution["ab_radius"]

        previous_center = (
            np.asarray(existing["lab_center"], dtype=np.float64)
            if existing is not None
            else None
        )
        previous_source_type = (
            existing.get("source_type") if existing is not None else None
        )
        previous_status = (
            existing.get("state_status", "stable")
            if existing is not None
            else None
        )
        previous_member_ids = (
            list(existing.get("member_image_ids", []))
            if existing is not None
            else []
        )

        values = {
            "source_type": str(source_type),
            "source_expert_id": (
                int(source_expert_id)
                if source_expert_id is not None
                else None
            ),
            "active_expert_id": (
                int(active_expert_id)
                if active_expert_id is not None
                else None
            ),
            "cluster_id": int(cluster_id) if cluster_id is not None else None,
            "lab_center": lab_center.copy(),
            "l_threshold": (
                float(l_threshold) if l_threshold is not None else None
            ),
            "ab_threshold": (
                float(ab_threshold) if ab_threshold is not None else None
            ),
            "dino_center": dino_center,
            "has_patch_memory": bool(has_patch_memory),
            "last_seen_step": int(step),
            "member_records": combined_records,
            "member_image_ids": (
                distribution["member_image_ids"]
                if distribution is not None
                else [int(record["image_id"]) for record in combined_records]
            ),
            "member_reliabilities": (
                distribution.get("member_reliabilities", [])
                if distribution is not None
                else [
                    float(record.get("member_reliability", 1.0))
                    for record in combined_records
                ]
            ),
            "l_radius": max(
                float(l_threshold or 0.0), self.photometric_l_ewma_floor
            ),
            "ab_radius": max(
                float(ab_threshold or 0.0), self.photometric_ab_ewma_floor
            ),
        }
        if temporary_support_evidence is not None:
            values.update(
                {
                    "temporary_support_increment": float(
                        temporary_support_evidence.get("increment", 0.0)
                    ),
                    "temporary_dino_evidence": float(
                        temporary_support_evidence.get(
                            "dino_evidence", 0.0
                        )
                    ),
                    "temporary_channel_evidence": float(
                        temporary_support_evidence.get(
                            "channel_evidence", 0.0
                        )
                    ),
                    "temporary_risk_evidence": float(
                        temporary_support_evidence.get(
                            "risk_evidence", 0.0
                        )
                    ),
                    "temporary_support_last_step": int(step),
                }
            )
        if existing is None:
            existing = {
                "state_id": int(self.next_historical_state_id),
                "created_step": int(step),
                "support_count": int(support_count),
                "support_event_count": max(
                    1, int(support_event_increment)
                ),
                "state_status": str(state_status),
                "last_center_shift": None,
                **values,
            }
            if (
                self.photometric_denstream_support_enabled
                and temporary_support_evidence is not None
            ):
                existing["temporary_effective_support"] = float(
                    (temporary_support_evidence or {}).get("increment", 0.0)
                )
            self.next_historical_state_id += 1
            self.historical_states.append(existing)
            self._record_state_lifecycle_event(
                "create",
                step,
                existing,
                reason="new_historical_state",
            )
            if (
                self.photometric_denstream_support_enabled
                and temporary_support_evidence is not None
                and state_status == "stable"
            ):
                self._record_state_lifecycle_event(
                    "mature",
                    step,
                    existing,
                    source_type_before="temporary",
                    status_before="temporary",
                    reason="denstream_effective_support_matured",
                )
        else:
            previous_count = int(existing.get("support_count", 0))
            previous_event_count = int(
                existing.get("support_event_count", 1)
            )
            previous_status = existing.get("state_status", "stable")
            previous_effective_support = float(
                existing.get("temporary_effective_support", 0.0)
            )
            existing.update(values)
            existing["support_count"] = (
                previous_count + int(support_count)
                if accumulate_support
                else max(previous_count, int(support_count))
            )
            existing["support_event_count"] = (
                previous_event_count + int(support_event_increment)
            )
            existing["state_status"] = (
                "stable" if previous_status == "stable" else str(state_status)
            )
            if (
                self.photometric_denstream_support_enabled
                and previous_status == "temporary"
            ):
                existing["temporary_effective_support"] = float(
                    previous_effective_support
                    + (temporary_support_evidence or {}).get(
                        "increment", 0.0
                    )
                )
            if previous_center is not None:
                existing["last_center_shift"] = self._lab_distances(
                    lab_center, previous_center
                )
            if (
                previous_source_type != existing.get("source_type")
                or previous_status != existing.get("state_status", "stable")
                or previous_member_ids != existing.get("member_image_ids", [])
                or int(support_event_increment) > 0
            ):
                self._record_state_lifecycle_event(
                    "update",
                    step,
                    existing,
                    source_type_before=previous_source_type,
                    status_before=previous_status,
                    reason=(
                        "denstream_effective_support"
                        if self.photometric_denstream_support_enabled
                        and previous_status == "temporary"
                        else "historical_state_evidence"
                    ),
                )
            if (
                previous_status == "temporary"
                and existing.get("state_status") == "stable"
            ):
                self._record_state_lifecycle_event(
                    "mature",
                    step,
                    existing,
                    source_type_before=previous_source_type,
                    status_before=previous_status,
                    reason=(
                        "denstream_effective_support_matured"
                        if self.photometric_denstream_support_enabled
                        else "second_independent_support_event"
                    ),
                )
        self._trim_historical_states(step)
        registered = next(
            (
                state
                for state in self.historical_states
                if int(state["state_id"]) == int(existing["state_id"])
            ),
            None,
        )
        if registered is not None:
            return registered
        matching_states = []
        for state in self.historical_states:
            profile = self._historical_states_match(existing, state)
            if profile is not None and profile["matched"]:
                matching_states.append(
                    (profile["joint_match_strength"], state)
                )
        if matching_states:
            return max(matching_states, key=lambda item: item[0])[1]
        return None

    def _register_expert_historical_state(
        self,
        expert,
        source_type,
        step,
        state_id=None,
    ):
        lab_state = expert.get("lab_state")
        if lab_state is None or lab_state.get("center") is None:
            return None
        cluster_id = expert.get("cluster_id")
        if state_id is None and cluster_id is not None:
            cluster_state = next(
                (
                    state
                    for state in self.historical_states
                    if state.get("cluster_id") is not None
                    and int(state["cluster_id"]) == int(cluster_id)
                ),
                None,
            )
            if cluster_state is not None:
                state_id = int(cluster_state["state_id"])
        dino_center = self._historical_state_dino_center(cluster_id)
        cluster_members = (
            self.clusters[int(cluster_id)].get("members", [])
            if cluster_id is not None
            and 0 <= int(cluster_id) < len(self.clusters)
            else []
        )
        return self._upsert_historical_state(
            source_type=source_type,
            step=step,
            lab_center=lab_state["center"],
            dino_center=dino_center,
            support_count=expert.get("support_count", 0),
            source_expert_id=expert.get("expert_id"),
            active_expert_id=(
                expert.get("expert_id") if source_type == "active" else None
            ),
            cluster_id=cluster_id,
            l_threshold=lab_state.get("l_threshold"),
            ab_threshold=lab_state.get("ab_threshold"),
            has_patch_memory=(dino_center is not None),
            state_id=state_id,
            member_records=self._historical_member_records(cluster_members),
            state_status="stable",
        )

    def _sync_historical_states(self, step=None):
        sync_step = self.last_step if step is None else int(step)
        for expert in self.active_experts:
            self._register_expert_historical_state(
                expert,
                "active",
                expert.get("last_supported_step", sync_step),
            )
        for expert in self.retired_experts:
            cluster_id = int(expert["cluster_id"])
            if (
                0 <= cluster_id < len(self.clusters)
                and self.clusters[cluster_id].get("active_expert_id")
                is not None
            ):
                continue
            self._register_expert_historical_state(
                expert,
                "retired",
                expert.get("deletion_step", sync_step),
            )

    @staticmethod
    def _historical_state_audit_item(state):
        return {
            "state_id": int(state["state_id"]),
            "source_type": state.get("source_type"),
            "source_expert_id": state.get("source_expert_id"),
            "active_expert_id": state.get("active_expert_id"),
            "cluster_id": state.get("cluster_id"),
            "has_patch_memory": bool(
                state.get("has_patch_memory", False)
            ),
            "created_step": int(state.get("created_step", -1)),
            "last_seen_step": int(state.get("last_seen_step", -1)),
            "support_count": int(state.get("support_count", 0)),
            "support_event_count": int(
                state.get("support_event_count", 1)
            ),
            "temporary_effective_support": (
                float(state["temporary_effective_support"])
                if state.get("temporary_effective_support") is not None
                else None
            ),
            "temporary_support_last_step": state.get(
                "temporary_support_last_step"
            ),
            "temporary_support_increment": (
                float(state["temporary_support_increment"])
                if state.get("temporary_support_increment") is not None
                else None
            ),
            "state_status": state.get("state_status", "stable"),
            "member_image_ids": list(state.get("member_image_ids", [])),
            "l_radius": float(
                state.get("l_radius")
                or state.get("l_threshold")
                or 0.0
            ),
            "ab_radius": float(
                state.get("ab_radius")
                or state.get("ab_threshold")
                or 0.0
            ),
            "last_center_shift": state.get("last_center_shift"),
            "lab_center": np.asarray(
                state["lab_center"], dtype=np.float64
            ).tolist(),
        }

    def _historical_state_audit(self):
        return [
            self._historical_state_audit_item(state)
            for state in self.historical_states
        ]

    def _candidate_risk_profile(self, candidate_members):
        ratios = []
        low_risk_count = 0
        for member in candidate_members:
            threshold = member.get("historical_ms_threshold")
            score = member.get("ms_score")
            if threshold is None or score is None or float(threshold) <= 0.0:
                continue
            ratio = float(score) / max(float(threshold), 1e-12)
            ratios.append(ratio)
            low_risk_count += int(ratio <= 1.0)
        median_ratio = float(np.median(ratios)) if ratios else None
        max_ratio = float(max(ratios)) if ratios else None
        acceptable = bool(
            len(ratios) == len(candidate_members)
            and low_risk_count >= 1
            and median_ratio <= self.photometric_history_median_risk_max
            and max_ratio <= self.photometric_history_extreme_risk_max
        )
        return {
            "evaluated_count": len(ratios),
            "low_risk_count": int(low_risk_count),
            "risk_ratios": ratios,
            "median_risk_ratio": median_ratio,
            "max_risk_ratio": max_ratio,
            "median_risk_max": self.photometric_history_median_risk_max,
            "extreme_risk_max": self.photometric_history_extreme_risk_max,
            "acceptable": acceptable,
        }

    def _temporary_support_evidence(
        self,
        candidate_members,
        risk_profile,
    ):
        embeddings = [
            member.get("embedding") for member in candidate_members
            if member.get("embedding") is not None
        ]
        if len(embeddings) >= 2:
            stacked = torch.stack(embeddings)
            similarities = stacked @ stacked.T
            pair_indices = torch.triu_indices(
                len(embeddings), len(embeddings), offset=1
            )
            mean_similarity = float(
                similarities[pair_indices[0], pair_indices[1]].mean()
            )
            dino_evidence = max(
                0.0,
                min(
                    1.0,
                    (
                        mean_similarity
                        - self.photometric_dino_similarity_floor
                    )
                    / max(
                        1.0 - self.photometric_dino_similarity_floor,
                        1e-12,
                    ),
                ),
            )
        else:
            mean_similarity = None
            dino_evidence = 0.0

        channel_values = []
        for member in candidate_members:
            support = member.get("channel_support")
            if support is None:
                support = member.get("provisional_channel_support")
            if support is not None and math.isfinite(float(support)):
                channel_values.append(float(support))
        complete_channel_evidence = bool(
            len(channel_values) == len(candidate_members)
            and channel_values
        )
        channel_evidence = (
            float(np.mean(np.clip(channel_values, 0.0, 1.0)))
            if complete_channel_evidence
            else 0.0
        )

        risk_ratios = risk_profile.get("risk_ratios", [])
        complete_risk_evidence = bool(
            len(risk_ratios) == len(candidate_members) and risk_ratios
        )
        risk_values = (
            [
                min(1.0, 1.0 / max(float(ratio), 1e-12))
                for ratio in risk_ratios
            ]
            if complete_risk_evidence
            else []
        )
        risk_evidence = (
            float(np.mean(risk_values)) if risk_values else 0.0
        )
        member_evidence = (
            [
                float(
                    dino_evidence
                    * min(1.0, max(0.0, channel_value))
                    * risk_value
                )
                for channel_value, risk_value in zip(
                    channel_values,
                    risk_values,
                )
            ]
            if complete_channel_evidence and complete_risk_evidence
            else []
        )
        increment = 0.0
        for evidence in member_evidence:
            increment = (
                self.photometric_denstream_decay * increment + evidence
            )
        return {
            "increment": float(increment),
            "dino_evidence": float(dino_evidence),
            "dino_similarity_mean": mean_similarity,
            "channel_evidence": float(channel_evidence),
            "risk_evidence": float(risk_evidence),
            "member_evidence": member_evidence,
        }

    def _decay_temporary_states(self, step):
        if not self.photometric_denstream_support_enabled:
            return
        for state in list(self.historical_states):
            if state.get("state_status", "stable") != "temporary":
                continue
            last_step = int(
                state.get(
                    "temporary_support_last_step",
                    state.get("last_seen_step", step),
                )
            )
            elapsed = max(0, int(step) - last_step)
            if elapsed == 0:
                continue
            previous_support = float(
                state.get("temporary_effective_support", 0.0)
            )
            state["temporary_effective_support"] = float(
                previous_support
                * self.photometric_denstream_decay ** elapsed
            )
            state["temporary_support_last_step"] = int(step)
            state["temporary_support_increment"] = 0.0
            if (
                state["temporary_effective_support"]
                < self.photometric_denstream_discard_support
            ):
                self.historical_states.remove(state)
                self._record_state_lifecycle_event(
                    "evict",
                    step,
                    state,
                    source_type_before=state.get("source_type"),
                    status_before="temporary",
                    reason="temporary_effective_support_decay",
                )
            else:
                self._record_state_lifecycle_event(
                    "decay",
                    step,
                    state,
                    source_type_before=state.get("source_type"),
                    status_before="temporary",
                    reason="temporary_effective_support_decay",
                )

    def _candidate_admission_evidence(
        self,
        candidate_members,
        photometric_profile,
        support_threshold,
    ):
        selected_ids = set(
            int(image_id)
            for image_id in photometric_profile.get(
                "selected_inlier_image_ids", []
            )
        )
        selected_members = [
            member
            for member in candidate_members
            if int(member["image_id"]) in selected_ids
        ]
        (
            new_coverage_fraction,
            new_coverage_count,
            evaluated_count,
        ) = self._candidate_new_coverage_fraction(
            selected_members,
            support_threshold,
        )
        channel_novel = bool(
            new_coverage_fraction is not None
            and new_coverage_fraction
            >= self.budget_min_new_coverage_fraction
        )
        has_reliable_patch_memory = bool(
            len(selected_members) >= 2
            and all(
                member.get("memory_patch_ids")
                for member in selected_members
            )
        )
        history_match = self._historical_state_match(
            selected_members
        )
        profile_consistent = bool(
            photometric_profile.get("consistent", False)
        )
        strict_low_risk_count = sum(
            bool(member.get("historical_low_risk", False))
            for member in selected_members
        )
        risk_profile = self._candidate_risk_profile(selected_members)
        temporary_support_evidence = self._temporary_support_evidence(
            selected_members,
            risk_profile,
        )
        history_match_strength = float(
            history_match.get("joint_match_strength", 0.0)
        )
        history_required_count = (
            self.photometric_history_reentry_members
            if history_match_strength >= 0.5
            else self.photometric_history_reentry_members + 1
        )
        matched_status = history_match.get("matched_profile", {}).get(
            "state_status"
        )
        matched_source_type = history_match.get("matched_source_type")
        if (
            profile_consistent
            and history_match.get("matched", False)
            and (
                matched_status == "temporary"
                or matched_source_type == "temporary"
            )
            and self.photometric_temporary_state_enabled
            and risk_profile["acceptable"]
        ):
            decision = "update_temporary_state"
        elif (
            profile_consistent
            and history_match.get("matched", False)
            and matched_source_type == "active"
            and self.photometric_state_distribution_enabled
            and channel_novel
            and has_reliable_patch_memory
            and risk_profile["acceptable"]
        ):
            decision = "update_existing_state"
        elif (
            profile_consistent
            and history_match.get("matched", False)
            and matched_source_type == "active"
            and self.photometric_state_distribution_enabled
            and risk_profile["acceptable"]
        ):
            decision = "existing_state_supported"
        elif (
            profile_consistent
            and history_match.get("matched", False)
            and matched_source_type == "photometric_only"
            and (
                self.photometric_history_reentry_enabled
                or self.photometric_state_distribution_enabled
            )
            and not (
                channel_novel and has_reliable_patch_memory
            )
            and risk_profile["acceptable"]
        ):
            decision = "update_existing_state"
        elif (
            profile_consistent
            and history_match.get("matched", False)
            and matched_source_type in {"retired", "photometric_only"}
            and (
                matched_source_type == "retired"
                or (channel_novel and has_reliable_patch_memory)
            )
            and len(selected_members) >= history_required_count
            and risk_profile["acceptable"]
        ):
            decision = "history_reentry"
        elif (
            profile_consistent
            and not history_match.get("matched", False)
            and self.photometric_temporary_state_enabled
            and risk_profile["acceptable"]
        ):
            decision = "new_temporary_state"
        elif (
            profile_consistent
            and not history_match.get("matched", False)
            and self.photometric_adaptive_candidate_evidence
            and channel_novel
            and has_reliable_patch_memory
            and risk_profile["acceptable"]
        ):
            decision = "new_scoring_expert"
        else:
            decision = "keep_collecting"
        return {
            "decision": decision,
            "selected_members": selected_members,
            "history_match": history_match,
            "profile_consistent": profile_consistent,
            "channel_novel": channel_novel,
            "has_reliable_patch_memory": has_reliable_patch_memory,
            "new_coverage_fraction": new_coverage_fraction,
            "new_coverage_count": int(new_coverage_count),
            "candidate_observation_count": int(evaluated_count),
            "strict_low_risk_count": strict_low_risk_count,
            "history_required_count": int(history_required_count),
            "risk_profile": risk_profile,
            "temporary_support_evidence": temporary_support_evidence,
        }

    def _record_photometric_state(
        self,
        members,
        step,
        *,
        state_id=None,
        state_status="stable",
        source_type="photometric_only",
        temporary_support_evidence=None,
    ):
        lab_center = self._photometric_pool_lab_center(members)
        if lab_center is None:
            return None
        existing_state = next(
            (
                state
                for state in self.historical_states
                if state_id is not None
                and int(state["state_id"]) == int(state_id)
            ),
            None,
        )
        embeddings = [
            member["embedding"]
            for member in members
            if member.get("embedding") is not None
        ]
        dino_center = (
            F.normalize(torch.stack(embeddings).sum(dim=0), dim=0)
            if embeddings
            else None
        )
        historical_state = self._upsert_historical_state(
            source_type=source_type,
            step=step,
            lab_center=lab_center,
            dino_center=dino_center,
            support_count=len(members),
            l_threshold=self.photometric_l_ewma_floor,
            ab_threshold=self.photometric_ab_ewma_floor,
            source_expert_id=(
                existing_state.get("source_expert_id")
                if existing_state is not None
                else None
            ),
            active_expert_id=(
                existing_state.get("active_expert_id")
                if existing_state is not None
                else None
            ),
            cluster_id=(
                existing_state.get("cluster_id")
                if existing_state is not None
                else None
            ),
            has_patch_memory=bool(
                existing_state is not None
                and existing_state.get("has_patch_memory", False)
            ),
            accumulate_support=True,
            state_id=state_id,
            member_records=self._historical_member_records(members),
            state_status=state_status,
            support_event_increment=1,
            temporary_support_evidence=temporary_support_evidence,
        )
        return (
            {
                **self._historical_state_audit_item(historical_state),
                "historical_state_id": int(historical_state["state_id"]),
            }
            if historical_state is not None
            else None
        )

    def _update_matched_historical_state(
        self,
        history_match,
        members,
        step,
        temporary_support_evidence=None,
    ):
        state_id = history_match.get("matched_state_id")
        state = next(
            (
                item
                for item in self.historical_states
                if state_id is not None
                and int(item["state_id"]) == int(state_id)
            ),
            None,
        )
        if state is None:
            return None

        status_before = state.get("state_status", "stable")
        source_type = state.get("source_type", "photometric_only")
        if status_before != "temporary":
            temporary_support_evidence = None
        if (
            self.photometric_denstream_support_enabled
            and status_before == "temporary"
        ):
            next_effective_support = float(
                state.get("temporary_effective_support", 0.0)
                + (temporary_support_evidence or {}).get("increment", 0.0)
            )
            next_status = (
                "stable"
                if next_effective_support
                >= self.photometric_denstream_mature_support
                else "temporary"
            )
        else:
            next_status = (
                "stable"
                if status_before == "stable"
                or int(state.get("support_event_count", 1)) + 1 >= 2
                else "temporary"
            )
        updated = self._record_photometric_state(
            members,
            step,
            state_id=state_id,
            state_status=next_status,
            source_type=(
                "photometric_only"
                if source_type == "temporary" and next_status == "stable"
                else source_type
            ),
            temporary_support_evidence=temporary_support_evidence,
        )
        if updated is None:
            return None
        state_id = updated["historical_state_id"]
        state = next(
            item
            for item in self.historical_states
            if int(item["state_id"]) == int(state_id)
        )

        active_expert_id = state.get("active_expert_id")
        members_added = 0
        if active_expert_id is not None:
            expert = next(
                (
                    item
                    for item in self.active_experts
                    if int(item["expert_id"]) == int(active_expert_id)
                ),
                None,
            )
            if expert is not None:
                cluster = self.clusters[int(expert["cluster_id"])]
                existing_keys = {
                    (int(item.get("step", -1)), int(item["image_id"]))
                    for item in cluster.get("members", [])
                }
                new_members = [
                    member
                    for member in members
                    if (
                        int(member.get("step", -1)),
                        int(member["image_id"]),
                    ) not in existing_keys
                ]
                if new_members:
                    cluster["members"].extend(new_members)
                    cluster["embedding_sum"] = (
                        cluster["embedding_sum"]
                        + torch.stack(
                            [member["embedding"] for member in new_members]
                        ).sum(dim=0)
                    )
                    members_added = len(new_members)
                lab_members = [
                    cluster_member
                    for cluster_member in cluster["members"]
                    if cluster_member.get("lab_feature") is not None
                ]
                if lab_members:
                    expert["lab_state"] = self._build_lab_state(
                        lab_members,
                        existing_state=expert.get("lab_state"),
                        freeze_anchor=True,
                        step=step,
                    )
                    if next_status == "stable":
                        self._rebase_lab_detector(
                            expert,
                            members,
                            step,
                        )
                state = self._register_expert_historical_state(
                    expert,
                    "active",
                    step,
                    state_id=state_id,
                )

        return {
            **(updated or {}),
            "state_status_before": status_before,
            "state_status_after": state.get("state_status", "stable"),
            "active_expert_id": active_expert_id,
            "members_added_to_active_cluster": int(members_added),
        }

    def _expert_coverage_profiles(self):
        active_ids = {
            int(expert["expert_id"])
            for expert in self.active_experts
        }
        profiles = {}
        for expert in self.active_experts:
            expert_id = int(expert["expert_id"])
            observed = 0
            covered = 0
            unique = 0
            redundant = 0
            for record in self.coverage_history:
                supports = record["supports"]
                if expert_id not in supports:
                    continue
                observed += 1
                threshold = float(record["support_threshold"])
                if supports[expert_id] < threshold:
                    continue
                covered += 1
                other_supported = any(
                    int(other_id) in active_ids
                    and int(other_id) != expert_id
                    and float(other_support) >= threshold
                    for other_id, other_support in supports.items()
                )
                if other_supported:
                    redundant += 1
                else:
                    unique += 1
            profiles[expert_id] = {
                "observed_count": int(observed),
                "covered_count": int(covered),
                "unique_count": int(unique),
                "redundant_count": int(redundant),
                "coverage_fraction": (
                    float(covered / observed) if observed else 0.0
                ),
                "unique_fraction": (
                    float(unique / observed) if observed else 1.0
                ),
                "redundant_fraction": (
                    float(redundant / observed) if observed else 0.0
                ),
            }
        return profiles

    def _candidate_expert_lab_distance_audit(self, candidate_members):
        candidate_center = self._photometric_pool_lab_center(
            candidate_members
        )
        member_count = sum(
            member.get("lab_feature") is not None
            for member in candidate_members
        )
        audit = {
            "candidate_lab_center": (
                candidate_center.tolist()
                if candidate_center is not None
                else None
            ),
            "candidate_lab_member_count": int(member_count),
            "expert_lab_distances": [],
            "lab_distance_missing_expert_ids": [],
            "nearest_lab_expert_id": None,
            "nearest_lab_total_distance": None,
        }
        if candidate_center is None:
            return audit

        for expert in self.active_experts:
            expert_id = int(expert["expert_id"])
            lab_state = expert.get("lab_state")
            if lab_state is None or lab_state.get("center") is None:
                audit["lab_distance_missing_expert_ids"].append(
                    expert_id
                )
                continue
            distances = self._lab_distances(
                candidate_center,
                np.asarray(lab_state["center"], dtype=np.float64),
            )
            l_threshold = lab_state.get("l_threshold")
            ab_threshold = lab_state.get("ab_threshold")
            l_threshold = (
                float(l_threshold) if l_threshold is not None else None
            )
            ab_threshold = (
                float(ab_threshold) if ab_threshold is not None else None
            )
            l_novelty_threshold = (
                max(l_threshold, self.photometric_l_ewma_floor)
                if l_threshold is not None
                else None
            )
            ab_novelty_threshold = (
                max(ab_threshold, self.photometric_ab_ewma_floor)
                if ab_threshold is not None
                else None
            )
            audit["expert_lab_distances"].append(
                {
                    "expert_id": expert_id,
                    "lifecycle_state": expert.get(
                        "lifecycle_state", "active"
                    ),
                    "admission_route": expert.get(
                        "admission_route", "legacy"
                    ),
                    "l_threshold": l_threshold,
                    "ab_threshold": ab_threshold,
                    "l_novelty_threshold": l_novelty_threshold,
                    "ab_novelty_threshold": ab_novelty_threshold,
                    "l_outside_baseline": (
                        distances["l_distance"] > l_novelty_threshold
                        if l_novelty_threshold is not None
                        else None
                    ),
                    "ab_outside_baseline": (
                        distances["ab_distance"] > ab_novelty_threshold
                        if ab_novelty_threshold is not None
                        else None
                    ),
                    **distances,
                }
            )

        if audit["expert_lab_distances"]:
            nearest = min(
                audit["expert_lab_distances"],
                key=lambda item: item["total_distance"],
            )
            audit["nearest_lab_expert_id"] = int(
                nearest["expert_id"]
            )
            audit["nearest_lab_total_distance"] = float(
                nearest["total_distance"]
            )
        return audit

    def _historical_state_match(self, candidate_members):
        audit = {
            "enabled": self.photometric_history_reentry_enabled,
            "required_count": self.photometric_history_reentry_members,
            "candidate_member_count": len(candidate_members),
            "matched": False,
            "matched_state_id": None,
            "matched_source_type": None,
            "matched_active_expert_id": None,
            "matched_retired_expert_id": None,
            "matched_retired_cluster_id": None,
            "state_profiles": [],
            "retired_state_profiles": [],
        }
        if (
            (
                not self.photometric_history_reentry_enabled
                and not self.photometric_state_distribution_enabled
                and not self.photometric_temporary_state_enabled
                and not self.photometric_adaptive_candidate_evidence
            )
            or len(candidate_members)
            < self.photometric_history_reentry_members
        ):
            return audit

        candidate_center = self._photometric_pool_lab_center(
            candidate_members
        )
        if candidate_center is None:
            return audit
        self._sync_historical_states()
        candidate_records = self._historical_member_records(candidate_members)
        candidate_distribution = self._historical_state_distribution(
            candidate_records
        )
        if candidate_distribution is None:
            return audit
        candidate_center = candidate_distribution["lab_center"]
        candidate_embedding = candidate_distribution["dino_center"]

        matches = []
        for state in self.historical_states:
            lab_center = state.get("lab_center")
            state_embedding = state.get("dino_center")
            if lab_center is None or state_embedding is None:
                continue
            dino_similarity = float(
                candidate_embedding @ state_embedding
            )
            distances = self._lab_distances(
                candidate_center,
                np.asarray(lab_center, dtype=np.float64),
            )
            state_l_radius = max(
                float(
                    state.get("l_radius")
                    or state.get("l_threshold")
                    or 0.0
                ),
                self.photometric_l_ewma_floor,
            )
            state_ab_radius = max(
                float(
                    state.get("ab_radius")
                    or state.get("ab_threshold")
                    or 0.0
                ),
                self.photometric_ab_ewma_floor,
            )
            l_threshold = state_l_radius + float(
                candidate_distribution["l_radius"]
            )
            ab_threshold = state_ab_radius + float(
                candidate_distribution["ab_radius"]
            )
            l_relative = distances["l_distance"] / l_threshold
            ab_relative = distances["ab_distance"] / ab_threshold
            matched = bool(
                l_relative <= 1.0
                and ab_relative <= 1.0
                and dino_similarity
                >= self.photometric_dino_similarity_floor
            )
            max_relative = max(l_relative, ab_relative)
            lab_strength = max(0.0, 1.0 - max_relative)
            dino_denominator = max(
                1.0 - self.photometric_dino_similarity_floor,
                1e-12,
            )
            dino_strength = min(
                1.0,
                max(
                    0.0,
                    (
                        dino_similarity
                        - self.photometric_dino_similarity_floor
                    )
                    / dino_denominator,
                ),
            )
            profile = {
                "state_id": int(state["state_id"]),
                "source_type": state.get("source_type"),
                "source_expert_id": state.get("source_expert_id"),
                "active_expert_id": state.get("active_expert_id"),
                "cluster_id": state.get("cluster_id"),
                "state_status": state.get("state_status", "stable"),
                "last_seen_step": state.get("last_seen_step"),
                "has_patch_memory": bool(
                    state.get("has_patch_memory", False)
                ),
                "dino_similarity": dino_similarity,
                "l_threshold": l_threshold,
                "ab_threshold": ab_threshold,
                "state_l_radius": state_l_radius,
                "state_ab_radius": state_ab_radius,
                "candidate_l_radius": float(
                    candidate_distribution["l_radius"]
                ),
                "candidate_ab_radius": float(
                    candidate_distribution["ab_radius"]
                ),
                "l_relative_distance": float(l_relative),
                "ab_relative_distance": float(ab_relative),
                "lab_match_strength": float(lab_strength),
                "dino_match_strength": float(dino_strength),
                "joint_match_strength": float(
                    (lab_strength + dino_strength) / 2.0
                ),
                "matched": matched,
                **distances,
            }
            audit["state_profiles"].append(profile)
            if state.get("source_type") == "retired":
                audit["retired_state_profiles"].append(profile)
            if matched:
                matches.append(profile)

        if not matches:
            return audit
        best = max(
            matches,
            key=lambda item: (
                item["joint_match_strength"],
                item["dino_similarity"],
                int(item.get("last_seen_step") or -1),
            ),
        )
        audit.update(
            {
                "matched": True,
                "matched_state_id": int(best["state_id"]),
                "matched_source_type": best.get("source_type"),
                "matched_active_expert_id": best.get("active_expert_id"),
                "matched_retired_expert_id": (
                    int(best["source_expert_id"])
                    if best.get("source_type") == "retired"
                    and best.get("source_expert_id") is not None
                    else None
                ),
                "matched_retired_cluster_id": (
                    int(best["cluster_id"])
                    if best.get("source_type") == "retired"
                    and best.get("cluster_id") is not None
                    else None
                ),
                "joint_match_strength": float(
                    best["joint_match_strength"]
                ),
                "matched_profile": dict(best),
            }
        )
        matched_state = next(
            state
            for state in self.historical_states
            if int(state["state_id"]) == int(best["state_id"])
        )
        self._record_state_lifecycle_event(
            "match",
            max(
                [int(member.get("step", self.last_step + 1))
                 for member in candidate_members],
                default=self.last_step + 1,
            ),
            matched_state,
            matched_or_merged_state_id=matched_state["state_id"],
            source_type_before=matched_state.get("source_type"),
            status_before=matched_state.get("state_status", "stable"),
            reason="candidate_distribution_match",
        )
        return audit

    def _retired_photometric_state_match(self, candidate_members):
        return self._historical_state_match(candidate_members)

    def _history_reentry_support_profile(
        self,
        candidate_members,
        support_threshold,
    ):
        active_ids = {
            int(expert["expert_id"])
            for expert in self.active_experts
            if expert.get("lifecycle_state", "active") == "active"
        }
        support_counts = {expert_id: 0 for expert_id in active_ids}
        evaluated_count = 0
        supported_count = 0
        for member in candidate_members:
            supports = {
                int(expert_id): float(support)
                for expert_id, support in member.get(
                    "current_expert_supports", {}
                ).items()
                if int(expert_id) in active_ids
            }
            if not supports or support_threshold is None:
                continue
            evaluated_count += 1
            member_supported = False
            for expert_id, support in supports.items():
                if support >= support_threshold:
                    support_counts[expert_id] += 1
                    member_supported = True
            supported_count += int(member_supported)

        member_count = len(candidate_members)
        return {
            "candidate_member_count": int(member_count),
            "evaluated_member_count": int(evaluated_count),
            "supported_member_count": int(supported_count),
            "existing_support_fraction": (
                float(supported_count / member_count)
                if member_count
                else 0.0
            ),
            "fully_evaluated": bool(
                member_count and evaluated_count == member_count
            ),
            "fully_covered": bool(
                member_count
                and evaluated_count == member_count
                and supported_count == member_count
            ),
            "expert_support_fractions": {
                str(expert_id): (
                    float(count / member_count) if member_count else 0.0
                )
                for expert_id, count in support_counts.items()
            },
        }

    def _select_history_reentry_victim(
        self,
        candidate_members,
        support_threshold,
        admission_route,
    ):
        if not self.active_experts:
            return None, None

        support_profile = self._history_reentry_support_profile(
            candidate_members,
            support_threshold,
        )
        support_fractions = support_profile["expert_support_fractions"]
        coverage_profiles = self._expert_coverage_profiles()
        lab_profiles = self._expert_lab_uniqueness_profiles()
        victim = min(
            self.active_experts,
            key=lambda expert: (
                expert.get("lifecycle_state", "active") != "dormant",
                support_fractions.get(str(int(expert["expert_id"])), 0.0),
                not lab_profiles[int(expert["expert_id"])][
                    "photometrically_redundant"
                ],
                coverage_profiles[int(expert["expert_id"])][
                    "unique_fraction"
                ],
                int(expert["last_supported_step"]),
                int(expert["support_count"]),
                int(expert["admission_step"]),
            ),
        )
        victim_id = int(victim["expert_id"])
        audit = {
            "admission_route": admission_route,
            "candidate_admission_evidence": "history_reentry",
            "replacement_reason": "history_reentry_swap",
            "outcome": "accepted",
            "victim_expert_id": victim_id,
            "victim_lifecycle_state": victim.get(
                "lifecycle_state", "active"
            ),
            "victim_coverage": dict(coverage_profiles[victim_id]),
            "victim_lab_uniqueness": dict(lab_profiles[victim_id]),
            "history_reentry_support": support_profile,
        }
        self.last_budget_replacement_evaluation = dict(audit)
        return victim_id, audit

    def _activate_historical_state(
        self,
        history_match,
        candidate_members,
        step,
        support_threshold,
        admission_route,
        candidate_evidence=None,
    ):
        state_id = history_match.get("matched_state_id")
        matched_state = next(
            (
                state
                for state in self.historical_states
                if int(state["state_id"]) == int(state_id)
            ),
            None,
        ) if state_id is not None else None
        retired_expert = (
            next(
                (
                    expert
                    for expert in self.retired_experts
                    if matched_state is not None
                    and matched_state.get("source_expert_id") is not None
                    and int(expert["expert_id"])
                    == int(matched_state["source_expert_id"])
                ),
                None,
            )
            if matched_state is not None
            else None
        )
        support_profile = self._history_reentry_support_profile(
            candidate_members,
            support_threshold,
        )
        event = {
            "matched_state_id": state_id,
            "matched_source_type": (
                matched_state.get("source_type")
                if matched_state is not None
                else None
            ),
            "matched_source_expert_id": (
                matched_state.get("source_expert_id")
                if matched_state is not None
                else None
            ),
            "matched_cluster_id": (
                matched_state.get("cluster_id")
                if matched_state is not None
                else None
            ),
            "candidate_image_ids": [
                int(member["image_id"]) for member in candidate_members
            ],
            "existing_support": support_profile,
            "restored_expert_id": None,
            "victim_expert_id": None,
        }
        if matched_state is None:
            event["decision"] = "history_reentry_rejected"
            event["reason"] = "missing_historical_state"
            self.last_history_reentry_event = event
            return None, None

        active_expert_id = matched_state.get("active_expert_id")
        if active_expert_id is not None or support_profile["fully_covered"]:
            matched_state["last_seen_step"] = int(step)
            matched_state["support_count"] += len(candidate_members)
            event["decision"] = "history_shared_coverage"
            event["reason"] = (
                "matched_state_already_active"
                if active_expert_id is not None
                else "existing_expert_coverage"
            )
            self.last_history_reentry_event = event
            return matched_state.get("cluster_id"), None

        cluster_id = matched_state.get("cluster_id")
        can_restore_cluster = bool(
            matched_state.get("has_patch_memory", False)
            and cluster_id is not None
            and 0 <= int(cluster_id) < len(self.clusters)
            and retired_expert is not None
        )
        if not can_restore_cluster:
            representative = min(
                candidate_members,
                key=lambda item: item["ms_short_score"],
            )
            replaced_before = self.last_budget_replacement
            historical_ids_before = {
                int(state["state_id"])
                for state in self.historical_states
            }
            new_cluster_id, expert_id = self._admit_quarantined_member(
                representative,
                step,
                cluster_members=candidate_members,
                admission_route=admission_route,
                support_threshold=support_threshold,
                candidate_evidence=candidate_evidence,
            )
            if expert_id is None:
                self._update_matched_historical_state(
                    history_match,
                    candidate_members,
                    step,
                )
                event["decision"] = "history_reentry_rejected"
                event["reason"] = (
                    (self.last_budget_replacement_evaluation or {}).get(
                        "outcome",
                        "historical_state_materialization_failed",
                    )
                )
                event["retained_in_history"] = True
                self.last_history_reentry_event = event
                return None, None
            new_expert = next(
                expert
                for expert in self.active_experts
                if int(expert["expert_id"]) == int(expert_id)
            )
            self.historical_states = [
                state
                for state in self.historical_states
                if int(state["state_id"]) in historical_ids_before
                or state.get("source_expert_id") != int(expert_id)
            ]
            self._register_expert_historical_state(
                new_expert,
                "active",
                step,
                state_id=state_id,
            )
            replaced_now = bool(
                self.last_budget_replacement is not None
                and self.last_budget_replacement is not replaced_before
                and int(self.last_budget_replacement.get("step", -1))
                == int(step)
            )
            event["decision"] = (
                "history_materialized_by_replacement"
                if replaced_now
                else "history_materialized_free_slot"
            )
            event["reason"] = "historical_state_materialized"
            event["restored_expert_id"] = int(expert_id)
            event["victim_expert_id"] = (
                int(self.last_budget_replacement["expert_id"])
                if replaced_now
                else None
            )
            self.last_history_reentry_event = event
            return new_cluster_id, expert_id

        cluster_id = int(cluster_id)
        cluster = self.clusters[cluster_id]
        restored_with_replacement = False
        if len(self.active_experts) >= self.committee_cap:
            victim_id, replacement_audit = (
                self._select_budget_replacement(
                    candidate_members,
                    support_threshold,
                    admission_route,
                    candidate_evidence=candidate_evidence,
                )
            )
            if (
                victim_id is None
                or not self._replace_budgeted_expert(
                    victim_id,
                    step,
                    replacement_audit,
                )
            ):
                self._update_matched_historical_state(
                    history_match,
                    candidate_members,
                    step,
                )
                event["decision"] = "history_reentry_rejected"
                event["reason"] = (
                    (self.last_budget_replacement_evaluation or {}).get(
                        "outcome", "replacement_failed"
                    )
                )
                event["retained_in_history"] = True
                self.last_history_reentry_event = event
                return None, None
            event["victim_expert_id"] = int(victim_id)
            restored_with_replacement = True

        existing_members = {
            (int(member["step"]), int(member["image_id"]))
            for member in cluster["members"]
        }
        for member in candidate_members:
            key = (int(member["step"]), int(member["image_id"]))
            if key in existing_members:
                continue
            cluster["members"].append(member)
            existing_members.add(key)
        cluster["embedding_sum"] = self._candidate_weighted_embedding_sum(
            cluster["members"]
        )

        representative = min(
            candidate_members,
            key=lambda item: item["ms_short_score"],
        )
        expert_id = self.next_expert_id
        restored_expert = {
            "expert_id": expert_id,
            "image_id": int(representative["image_id"]),
            "cluster_id": cluster_id,
            "admission_step": int(step),
            "cluster_support_at_admission": len(candidate_members),
            "max_support_gap": 0,
            "last_supported_step": int(step),
            "support_count": len(candidate_members),
            "patience": self.base_ttl,
            "ttl": self.base_ttl,
            "deletion_step": None,
            "last_event": "history_reentry",
            "lifecycle_state": "active",
            "last_channel_support": None,
            "representative_mode": self.representative_mode,
            "admission_ttl_mode": self.admission_ttl_mode,
            "admission_route": admission_route,
            "lab_state": self._build_lab_state(
                candidate_members,
                step=step,
            ),
            "admission_candidate_ms_score": representative["ms_score"],
            "admission_candidate_channel_support": representative[
                "channel_support"
            ],
            "reentry_from_expert_id": int(retired_expert["expert_id"]),
            "reentry_count": int(retired_expert.get("reentry_count", 0)) + 1,
        }
        if self.candidate_reliability_enabled:
            restored_expert["member_reliability"] = float(
                representative.get("member_reliability", 1.0)
            )
            restored_expert["scoring_patch_reliabilities"] = [
                float(value)
                for value in torch.as_tensor(
                    representative.get(
                        "scoring_patch_reliabilities",
                        representative["patch_reliabilities"],
                    ),
                    dtype=torch.float32,
                ).tolist()
            ]
            restored_expert["reference_reliability_penalty"] = float(
                self.reference_reliability_penalty
            )
            restored_expert["reliability_audit"] = dict(
                representative.get("reliability_audit", {})
            )
        cluster["active_expert_id"] = expert_id
        self.active_experts.append(restored_expert)
        self._register_expert_historical_state(
            restored_expert,
            "active",
            step,
            state_id=state_id,
        )
        self.admitted_count += 1
        self.next_expert_id += 1
        event["decision"] = (
            "history_restored_by_replacement"
            if restored_with_replacement
            else "history_restored_free_slot"
        )
        event["reason"] = "retired_state_restored"
        event["restored_expert_id"] = int(expert_id)
        self.last_history_reentry_event = event
        return cluster_id, expert_id

    def _candidate_lab_novelty(self, audit):
        distances = list(audit.get("expert_lab_distances", []))
        active_count = len(self.active_experts)
        comparable_to_all = bool(
            active_count
            and len(distances) == active_count
            and not audit.get("lab_distance_missing_expert_ids")
        )
        novel_to_all = bool(
            comparable_to_all
            and all(
                item.get("l_outside_baseline") is True
                or item.get("ab_outside_baseline") is True
                for item in distances
            )
        )
        return {
            "lab_comparable_to_all_experts": comparable_to_all,
            "lab_novel_to_all_experts": novel_to_all,
        }

    def _expert_lab_uniqueness_profiles(self):
        profiles = {}
        for expert in self.active_experts:
            expert_id = int(expert["expert_id"])
            state = expert.get("lab_state")
            nearest = None
            if state is not None and state.get("center") is not None:
                center = np.asarray(state["center"], dtype=np.float64)
                for other in self.active_experts:
                    other_id = int(other["expert_id"])
                    if other_id == expert_id:
                        continue
                    other_state = other.get("lab_state")
                    if (
                        other_state is None
                        or other_state.get("center") is None
                    ):
                        continue
                    distances = self._lab_distances(
                        center,
                        np.asarray(
                            other_state["center"], dtype=np.float64
                        ),
                    )
                    l_scale = max(
                        float(state.get("l_threshold") or 0.0),
                        float(other_state.get("l_threshold") or 0.0),
                        self.photometric_l_ewma_floor,
                    )
                    ab_scale = max(
                        float(state.get("ab_threshold") or 0.0),
                        float(other_state.get("ab_threshold") or 0.0),
                        self.photometric_ab_ewma_floor,
                    )
                    relative_distance = max(
                        distances["l_distance"] / l_scale,
                        distances["ab_distance"] / ab_scale,
                    )
                    item = {
                        "expert_id": other_id,
                        "relative_distance": float(relative_distance),
                        **distances,
                    }
                    if (
                        nearest is None
                        or item["relative_distance"]
                        < nearest["relative_distance"]
                    ):
                        nearest = item
            profiles[expert_id] = {
                "has_lab_state": bool(
                    state is not None and state.get("center") is not None
                ),
                "nearest_expert_id": (
                    int(nearest["expert_id"])
                    if nearest is not None
                    else None
                ),
                "nearest_relative_distance": (
                    float(nearest["relative_distance"])
                    if nearest is not None
                    else None
                ),
                "nearest_l_distance": (
                    float(nearest["l_distance"])
                    if nearest is not None
                    else None
                ),
                "nearest_ab_distance": (
                    float(nearest["ab_distance"])
                    if nearest is not None
                    else None
                ),
                "photometrically_redundant": bool(
                    nearest is not None
                    and nearest["relative_distance"] <= 1.0
                ),
            }
        return profiles

    def _coverage_loss_expert_entity(self, expert):
        cluster = self.clusters[int(expert["cluster_id"])]
        embedding_sum = cluster.get("embedding_sum")
        embedding = (
            F.normalize(embedding_sum.detach().cpu(), dim=0)
            if embedding_sum is not None
            and float(embedding_sum.norm()) > 0.0
            else None
        )
        lab_state = expert.get("lab_state") or {}
        center = lab_state.get("center")
        prototype_sets = [
            member.get("patch_prototypes")
            for member in cluster.get("members", [])
            if member.get("patch_prototypes") is not None
        ]
        return {
            "entity_id": int(expert["expert_id"]),
            "expert_id": int(expert["expert_id"]),
            "embedding": embedding,
            "patch_prototypes": (
                self._patch_coreset(torch.cat(prototype_sets, dim=0))
                if prototype_sets
                else None
            ),
            "lab_center": (
                np.asarray(center, dtype=np.float64)
                if center is not None
                else None
            ),
            "lab_l_scale": max(
                float(lab_state.get("l_threshold") or 0.0),
                self.photometric_l_ewma_floor,
            ),
            "lab_ab_scale": max(
                float(lab_state.get("ab_threshold") or 0.0),
                self.photometric_ab_ewma_floor,
            ),
            "candidate_image_ids": set(),
        }

    def _coverage_loss_candidate_entity(self, candidate_members):
        embeddings = [
            member.get("embedding")
            for member in candidate_members
            if member.get("embedding") is not None
        ]
        embedding = (
            F.normalize(torch.stack(embeddings).mean(dim=0), dim=0)
            if embeddings
            else None
        )
        prototype_sets = [
            member.get("patch_prototypes")
            for member in candidate_members
            if member.get("patch_prototypes") is not None
        ]
        lab_features = [
            np.asarray(member["lab_feature"], dtype=np.float64)
            for member in candidate_members
            if member.get("lab_feature") is not None
        ]
        lab_center = (
            np.median(np.stack(lab_features, axis=0), axis=0)
            if lab_features
            else None
        )
        l_scale = self.photometric_l_ewma_floor
        ab_scale = self.photometric_ab_ewma_floor
        if lab_center is not None and len(lab_features) >= 2:
            distances = [
                self._lab_distances(feature, lab_center)
                for feature in lab_features
            ]
            l_scale = max(
                float(np.median([item["l_distance"] for item in distances])),
                l_scale,
            )
            ab_scale = max(
                float(np.median([item["ab_distance"] for item in distances])),
                ab_scale,
            )
        return {
            "entity_id": "candidate",
            "expert_id": None,
            "embedding": embedding,
            "patch_prototypes": (
                self._patch_coreset(torch.cat(prototype_sets, dim=0))
                if prototype_sets
                else None
            ),
            "lab_center": lab_center,
            "lab_l_scale": float(l_scale),
            "lab_ab_scale": float(ab_scale),
            "candidate_image_ids": {
                int(member["image_id"]) for member in candidate_members
            },
        }

    def _coverage_loss_reference_records(
        self, candidate_members, support_threshold
    ):
        records = {}
        for record in self.coverage_history:
            image_id = int(record.get("image_id", record["step"]))
            records[image_id] = {
                "image_id": image_id,
                "step": int(record["step"]),
                "support_threshold": float(record["support_threshold"]),
                "supports": dict(record.get("supports", {})),
                "embedding": record.get("embedding"),
                "patch_prototypes": record.get("patch_prototypes"),
                "lab_feature": record.get("lab_feature"),
                "candidate_member": False,
            }
        for member in candidate_members:
            image_id = int(member["image_id"])
            existing = records.get(image_id, {})
            records[image_id] = {
                "image_id": image_id,
                "step": int(member.get("step", existing.get("step", 0))),
                "support_threshold": member.get(
                    "support_threshold",
                    existing.get("support_threshold", support_threshold),
                ),
                "supports": dict(
                    member.get(
                        "current_expert_supports",
                        existing.get("supports", {}),
                    )
                ),
                "embedding": member.get(
                    "embedding", existing.get("embedding")
                ),
                "patch_prototypes": member.get(
                    "patch_prototypes",
                    existing.get("patch_prototypes"),
                ),
                "lab_feature": member.get(
                    "lab_feature", existing.get("lab_feature")
                ),
                "candidate_member": True,
            }
        return sorted(records.values(), key=lambda item: item["step"])

    def _coverage_loss_objective(self, records, entities):
        channel_values = []
        lab_values = []
        dino_values = []
        for record in records:
            channel_scores = []
            lab_scores = []
            dino_scores = []
            patch_dino_scores = []
            record_prototypes = record.get("patch_prototypes")
            for entity in entities:
                expert_id = entity["expert_id"]
                if expert_id is None:
                    channel_score = float(
                        int(record["image_id"])
                        in entity["candidate_image_ids"]
                    )
                else:
                    threshold = record.get("support_threshold")
                    support = record.get("supports", {}).get(expert_id)
                    channel_score = (
                        min(max(float(support) / max(float(threshold), 1e-12), 0.0), 1.0)
                        if support is not None and threshold is not None
                        else 0.0
                    )
                channel_scores.append(channel_score)

                lab_feature = record.get("lab_feature")
                if lab_feature is None or entity["lab_center"] is None:
                    lab_scores.append(0.0)
                else:
                    distances = self._lab_distances(
                        np.asarray(lab_feature, dtype=np.float64),
                        entity["lab_center"],
                    )
                    relative = max(
                        distances["l_distance"] / entity["lab_l_scale"],
                        distances["ab_distance"] / entity["lab_ab_scale"],
                    )
                    lab_scores.append(float(math.exp(-relative)))

                embedding = record.get("embedding")
                if (
                    record_prototypes is not None
                    and entity.get("patch_prototypes") is not None
                ):
                    similarities = (
                        record_prototypes
                        @ entity["patch_prototypes"].T
                    )
                    patch_dino_scores.append(
                        ((similarities.max(dim=1).values + 1.0) / 2.0)
                        .clamp(0.0, 1.0)
                    )
                elif embedding is None or entity["embedding"] is None:
                    dino_scores.append(0.0)
                    if record_prototypes is not None:
                        patch_dino_scores.append(
                            torch.zeros(
                                int(record_prototypes.shape[0]),
                                dtype=torch.float32,
                            )
                        )
                else:
                    similarity = float(
                        F.normalize(embedding, dim=0)
                        @ entity["embedding"]
                    )
                    score = min(
                        max((similarity + 1.0) / 2.0, 0.0), 1.0
                    )
                    dino_scores.append(score)
                    if record_prototypes is not None:
                        patch_dino_scores.append(
                            torch.full(
                                (int(record_prototypes.shape[0]),),
                                score,
                                dtype=torch.float32,
                            )
                        )
            channel_values.append(max(channel_scores, default=0.0))
            lab_values.append(max(lab_scores, default=0.0))
            if patch_dino_scores:
                dino_values.append(
                    float(
                        torch.stack(patch_dino_scores)
                        .max(dim=0)
                        .values
                        .mean()
                    )
                )
            else:
                dino_values.append(max(dino_scores, default=0.0))

        modality_scores = {
            "channel": float(np.mean(channel_values)) if channel_values else 0.0,
            "lab": float(np.mean(lab_values)) if lab_values else 0.0,
            "dino": float(np.mean(dino_values)) if dino_values else 0.0,
        }
        weights = {
            "channel": self.budget_coverage_channel_weight,
            "lab": self.budget_coverage_lab_weight,
            "dino": self.budget_coverage_dino_weight,
        }
        weight_sum = sum(weights.values())
        total = sum(
            modality_scores[name] * weights[name] for name in weights
        ) / weight_sum
        return {
            "total": float(total),
            "channel": modality_scores["channel"],
            "lab": modality_scores["lab"],
            "dino": modality_scores["dino"],
            "observation_count": int(len(records)),
        }

    def _coverage_loss_replacement_profiles(
        self, candidate_members, support_threshold
    ):
        records = self._coverage_loss_reference_records(
            candidate_members, support_threshold
        )
        expert_entities = {
            int(expert["expert_id"]): self._coverage_loss_expert_entity(expert)
            for expert in self.active_experts
        }
        candidate_entity = self._coverage_loss_candidate_entity(
            candidate_members
        )
        current_entities = list(expert_entities.values())
        current = self._coverage_loss_objective(records, current_entities)
        profiles = {}
        for expert_id in expert_entities:
            retained = [
                entity
                for other_id, entity in expert_entities.items()
                if other_id != expert_id
            ]
            without = self._coverage_loss_objective(records, retained)
            with_candidate = self._coverage_loss_objective(
                records, retained + [candidate_entity]
            )
            profiles[expert_id] = {
                "current_coverage": dict(current),
                "coverage_without_expert": dict(without),
                "coverage_with_candidate": dict(with_candidate),
                "removal_loss": float(current["total"] - without["total"]),
                "swap_gain": float(with_candidate["total"] - current["total"]),
            }
        return profiles

    def _coverage_loss_joint_eviction_profiles(
        self, candidate_members, support_threshold
    ):
        records = self._coverage_loss_reference_records(
            candidate_members, support_threshold
        )
        entities = {
            int(expert["expert_id"]): self._coverage_loss_expert_entity(
                expert
            )
            for expert in self.active_experts
        }
        entities["candidate"] = self._coverage_loss_candidate_entity(
            candidate_members
        )
        joint = self._coverage_loss_objective(
            records, list(entities.values())
        )
        profiles = {}
        for entity_id in entities:
            retained = [
                entity
                for other_id, entity in entities.items()
                if other_id != entity_id
            ]
            retained_coverage = self._coverage_loss_objective(
                records, retained
            )
            profiles[entity_id] = {
                "joint_coverage": dict(joint),
                "coverage_after_eviction": dict(retained_coverage),
                "eviction_loss": float(
                    joint["total"] - retained_coverage["total"]
                ),
            }
        return profiles

    def _patch_kcenter_eviction_profiles(
        self, candidate_members, support_threshold
    ):
        records = self._coverage_loss_reference_records(
            candidate_members, support_threshold
        )
        entities = {
            int(expert["expert_id"]): self._coverage_loss_expert_entity(
                expert
            )
            for expert in self.active_experts
        }
        entities["candidate"] = self._coverage_loss_candidate_entity(
            candidate_members
        )

        def covering_radius(selected_entities):
            distances = []
            for record in records:
                record_prototypes = record.get("patch_prototypes")
                if record_prototypes is None:
                    continue
                similarities = []
                for entity in selected_entities:
                    prototypes = entity.get("patch_prototypes")
                    if prototypes is None:
                        continue
                    similarities.append(
                        record_prototypes @ prototypes.T
                    )
                if not similarities:
                    continue
                best_similarity = torch.stack(
                    [values.max(dim=1).values for values in similarities]
                ).max(dim=0).values
                distances.append(
                    (1.0 - best_similarity).clamp(0.0, 2.0)
                )
            if not distances:
                return 0.0
            return float(torch.cat(distances).max().item())

        joint_radius = covering_radius(list(entities.values()))
        profiles = {}
        for entity_id in entities:
            retained = [
                entity
                for other_id, entity in entities.items()
                if other_id != entity_id
            ]
            radius_after_eviction = covering_radius(retained)
            profiles[entity_id] = {
                "joint_covering_radius": float(joint_radius),
                "covering_radius_after_eviction": float(
                    radius_after_eviction
                ),
                "eviction_loss": float(
                    radius_after_eviction - joint_radius
                ),
                "coverage_objective": "patch_kcenter_max_cosine_distance",
            }
        return profiles

    def _virtual_entity_support(self, record, entity):
        """Estimate scoring-task support for a virtual committee member."""
        expert_id = entity.get("expert_id")
        if expert_id is None:
            channel_score = 0.0
        else:
            threshold = record.get("support_threshold")
            support = record.get("supports", {}).get(expert_id)
            channel_score = (
                min(
                    max(float(support) / max(float(threshold), 1e-12), 0.0),
                    1.0,
                )
                if support is not None and threshold is not None
                else 0.0
            )

        lab_feature = record.get("lab_feature")
        lab_center = entity.get("lab_center")
        if lab_feature is None or lab_center is None:
            lab_score = 0.0
        else:
            distances = self._lab_distances(
                np.asarray(lab_feature, dtype=np.float64), lab_center
            )
            relative = max(
                distances["l_distance"]
                / max(float(entity.get("lab_l_scale") or 0.0),
                      self.photometric_l_ewma_floor),
                distances["ab_distance"]
                / max(float(entity.get("lab_ab_scale") or 0.0),
                      self.photometric_ab_ewma_floor),
            )
            lab_score = float(math.exp(-relative))

        record_prototypes = record.get("patch_prototypes")
        entity_prototypes = entity.get("patch_prototypes")
        embedding = record.get("embedding")
        entity_embedding = entity.get("embedding")
        if record_prototypes is not None and entity_prototypes is not None:
            similarities = record_prototypes @ entity_prototypes.T
            dino_score = float(
                ((similarities.max(dim=1).values + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .mean()
            )
        elif embedding is not None and entity_embedding is not None:
            similarity = float(
                F.normalize(embedding, dim=0) @ entity_embedding
            )
            dino_score = min(max((similarity + 1.0) / 2.0, 0.0), 1.0)
        else:
            dino_score = 0.0

        if record_prototypes is not None and entity_prototypes is not None:
            return dino_score
        if embedding is not None and entity_embedding is not None:
            return dino_score
        if channel_score > 0.0:
            return channel_score
        return lab_score

    def _historical_revisit_records(self):
        records = []
        for state in self.historical_states:
            state_records = list(state.get("member_records", []))
            if not state_records:
                state_records = [
                    {
                        "step": int(state.get("last_seen_step", self.last_step)),
                        "image_id": -int(state["state_id"]) - 1,
                        "lab_feature": state.get("lab_center"),
                        "embedding": state.get("dino_center"),
                    }
                ]
            for record in state_records:
                if (
                    record.get("lab_feature") is None
                    or record.get("embedding") is None
                ):
                    continue
                records.append(
                    {
                        "state_id": int(state["state_id"]),
                        "step": int(record.get("step", self.last_step)),
                        "image_id": int(record.get("image_id", -1)),
                        "support_threshold": None,
                        "supports": {},
                        "embedding": record.get("embedding"),
                        "patch_prototypes": None,
                        "lab_feature": record.get("lab_feature"),
                    }
                )
        return records

    def _virtual_candidate_records(self, candidate_members, support_threshold):
        records = []
        for member in candidate_members:
            embedding = member.get("embedding")
            patch_prototypes = member.get("patch_prototypes")
            if embedding is None and patch_prototypes is None:
                continue
            records.append(
                {
                    "step": int(member.get("step", self.last_step)),
                    "image_id": int(member["image_id"]),
                    "support_threshold": support_threshold,
                    "supports": dict(
                        member.get("current_expert_supports", {})
                    ),
                    "embedding": embedding,
                    "patch_prototypes": patch_prototypes,
                    "lab_feature": member.get("lab_feature"),
                }
            )
        return records

    def _virtual_swap_profiles(
        self, candidate_members, support_threshold, candidate_evidence=None
    ):
        """Evaluate each victim on trusted history before committing a swap."""
        records = list(self.coverage_history)[
            -self.budget_virtual_swap_history_size :
        ]
        if not records or not self.active_experts:
            return {}

        expert_entities = {
            int(expert["expert_id"]): self._coverage_loss_expert_entity(expert)
            for expert in self.active_experts
        }
        candidate_entity = self._coverage_loss_candidate_entity(
            candidate_members
        )
        baseline_entities = list(expert_entities.values())
        baseline_support = np.asarray(
            [
                max(
                    self._virtual_entity_support(record, entity)
                    for entity in baseline_entities
                )
                for record in records
            ],
            dtype=np.float64,
        )
        baseline_scores = 1.0 - baseline_support
        baseline_iqr = float(
            np.percentile(baseline_scores, 75)
            - np.percentile(baseline_scores, 25)
        )
        revisit_records = self._historical_revisit_records()
        candidate_records = self._virtual_candidate_records(
            candidate_members, support_threshold
        )
        baseline_revisit_support = np.asarray(
            [
                max(
                    self._virtual_entity_support(record, entity)
                    for entity in baseline_entities
                )
                for record in revisit_records
            ],
            dtype=np.float64,
        )
        profiles = {}
        for victim_id in expert_entities:
            retained = [
                entity
                for entity_id, entity in expert_entities.items()
                if entity_id != victim_id
            ]
            swapped_entities = retained + [candidate_entity]
            swapped_support = np.asarray(
                [
                    max(
                        self._virtual_entity_support(record, entity)
                        for entity in swapped_entities
                    )
                    for record in records
                ],
                dtype=np.float64,
            )
            swapped_scores = 1.0 - swapped_support
            score_delta = swapped_scores - baseline_scores
            swapped_iqr = float(
                np.percentile(swapped_scores, 75)
                - np.percentile(swapped_scores, 25)
            )
            current_count = min(8, len(records))
            recent_support_loss = float(
                np.mean(
                    np.maximum(
                        baseline_support[-current_count:]
                        - swapped_support[-current_count:],
                        0.0,
                    )
                )
            )
            if candidate_records:
                baseline_candidate_support = np.asarray(
                    [
                        max(
                            self._virtual_entity_support(record, entity)
                            for entity in baseline_entities
                        )
                        for record in candidate_records
                    ],
                    dtype=np.float64,
                )
                swapped_candidate_support = np.asarray(
                    [
                        max(
                            self._virtual_entity_support(record, entity)
                            for entity in swapped_entities
                        )
                        for record in candidate_records
                    ],
                    dtype=np.float64,
                )
                current_support_gain = float(
                    np.mean(
                        np.maximum(
                            swapped_candidate_support
                            - baseline_candidate_support,
                            0.0,
                        )
                    )
                )
                current_support_loss = float(
                    np.mean(
                        np.maximum(
                            baseline_candidate_support
                            - swapped_candidate_support,
                            0.0,
                        )
                    )
                )
            else:
                current_support_gain = 0.0
                current_support_loss = 0.0
            if revisit_records:
                swapped_revisit_support = np.asarray(
                    [
                        max(
                            self._virtual_entity_support(record, entity)
                            for entity in swapped_entities
                        )
                        for record in revisit_records
                    ],
                    dtype=np.float64,
                )
                recurrent_loss = float(
                    np.mean(
                        np.maximum(
                            baseline_revisit_support
                            - swapped_revisit_support,
                            0.0,
                        )
                    )
                )
            else:
                recurrent_loss = 0.0
            median_shift = float(np.median(score_delta))
            mean_shift = float(np.mean(score_delta))
            iqr_increase = float(max(0.0, swapped_iqr - baseline_iqr))
            support_loss = float(
                np.mean(np.maximum(baseline_support - swapped_support, 0.0))
            )
            task_cost = float(
                max(
                    support_loss,
                    recent_support_loss,
                    current_support_loss,
                    recurrent_loss,
                )
                + iqr_increase
            )
            net_task_gain = float(current_support_gain - task_cost)
            safe = bool(
                mean_shift
                <= self.budget_virtual_swap_max_normal_score_shift
                and iqr_increase
                <= self.budget_virtual_swap_max_iqr_increase
                and current_support_loss
                <= self.budget_virtual_swap_max_normal_score_shift
                and recurrent_loss
                <= self.budget_virtual_swap_max_normal_score_shift
            )
            profiles[int(victim_id)] = {
                "history_count": int(len(records)),
                "median_normal_score_shift": median_shift,
                "mean_normal_score_shift": mean_shift,
                "normal_score_iqr_increase": iqr_increase,
                "normal_support_loss": support_loss,
                "current_state_support_loss": current_support_loss,
                "current_state_support_gain": current_support_gain,
                "recent_normal_support_loss": recent_support_loss,
                "recurrent_state_loss": recurrent_loss,
                "historical_revisit_count": int(len(revisit_records)),
                "task_cost": task_cost,
                "net_task_gain": net_task_gain,
                "max_normal_score_shift": float(np.max(score_delta)),
                "safe": safe,
            }
        return profiles

    def _expert_contribution_profiles(
        self,
        coverage_profiles,
        lab_uniqueness_profiles,
        reference_step,
    ):
        profiles = {}
        for expert in self.active_experts:
            expert_id = int(expert["expert_id"])
            coverage = coverage_profiles[expert_id]
            lab = lab_uniqueness_profiles[expert_id]
            channel_score = float(coverage["unique_fraction"])
            nearest_relative = lab.get("nearest_relative_distance")
            lab_score = (
                min(float(nearest_relative), 1.0)
                if nearest_relative is not None
                else 1.0
                if lab.get("has_lab_state", False)
                else 0.0
            )
            age = max(
                0,
                int(reference_step) - int(expert["last_supported_step"]),
            )
            recent_score = max(
                0.0,
                1.0 - age / max(float(self.budget_coverage_window), 1.0),
            )
            if expert.get("lifecycle_state", "active") == "dormant":
                recent_score = 0.0
            contribution = (
                channel_score + lab_score + recent_score
            ) / 3.0
            profiles[expert_id] = {
                "channel_unique_score": channel_score,
                "lab_unique_score": lab_score,
                "recent_support_score": recent_score,
                "contribution_score": float(contribution),
            }
        return profiles

    def _candidate_contribution_profile(
        self,
        evaluation,
        candidate_evidence,
    ):
        new_coverage = evaluation.get("new_coverage_fraction")
        channel_score = (
            min(max(float(new_coverage), 0.0), 1.0)
            if new_coverage is not None
            else 0.0
        )
        relative_distances = []
        for item in evaluation.get("expert_lab_distances", []):
            l_threshold = item.get("l_novelty_threshold")
            ab_threshold = item.get("ab_novelty_threshold")
            if l_threshold is None or ab_threshold is None:
                continue
            relative_distances.append(
                max(
                    float(item["l_distance"]) / max(float(l_threshold), 1e-12),
                    float(item["ab_distance"]) / max(float(ab_threshold), 1e-12),
                )
            )
        lab_score = (
            min(min(relative_distances), 1.0)
            if relative_distances
            else 0.0
        )
        history_match = dict(candidate_evidence or {}).get(
            "history_match", {}
        )
        history_score = (
            min(
                max(
                    float(history_match.get("joint_match_strength", 0.0)),
                    0.0,
                ),
                1.0,
            )
            if history_match.get("matched", False)
            else 0.0
        )
        reliability_score = history_score if history_score > 0.0 else 1.0
        contribution = (
            channel_score + lab_score + reliability_score
        ) / 3.0
        return {
            "channel_new_coverage_score": channel_score,
            "lab_unique_score": lab_score,
            "history_match_score": history_score,
            "candidate_reliability_score": reliability_score,
            "contribution_score": float(contribution),
        }

    def _select_budget_replacement(
        self,
        candidate_members,
        support_threshold,
        admission_route,
        candidate_evidence=None,
    ):
        if admission_route == "legacy":
            replacement_enabled = self.legacy_budget_replacement
        elif admission_route.startswith("photometric_drift"):
            replacement_enabled = self.photometric_budget_replacement
        else:
            replacement_enabled = False
        evaluation = {
            "admission_route": admission_route,
            "replacement_enabled": bool(replacement_enabled),
            "photometric_budget_admission_mode": (
                self.photometric_budget_admission_mode
            ),
            "minimum_new_coverage_fraction": float(
                self.budget_min_new_coverage_fraction
            ),
            "replacement_margin": float(self.budget_replacement_margin),
        }
        photometric_route = admission_route.startswith("photometric_drift")
        if photometric_route:
            evaluation.update(
                self._candidate_expert_lab_distance_audit(
                    candidate_members
                )
            )
            evaluation.update(self._candidate_lab_novelty(evaluation))
        if not replacement_enabled:
            evaluation["outcome"] = "replacement_disabled"
            self.last_budget_replacement_evaluation = evaluation
            return None, None

        if candidate_evidence is None:
            (
                new_coverage_fraction,
                new_coverage_count,
                candidate_observation_count,
            ) = self._candidate_new_coverage_fraction(
                candidate_members,
                support_threshold,
            )
        else:
            new_coverage_fraction = candidate_evidence.get(
                "new_coverage_fraction"
            )
            new_coverage_count = int(
                candidate_evidence.get("new_coverage_count", 0)
            )
            candidate_observation_count = int(
                candidate_evidence.get("candidate_observation_count", 0)
            )
        evaluation.update(
            {
                "new_coverage_fraction": new_coverage_fraction,
                "new_coverage_count": int(new_coverage_count),
                "candidate_observation_count": int(
                    candidate_observation_count
                ),
            }
        )
        channel_novel = bool(
            new_coverage_fraction is not None
            and new_coverage_fraction
            >= self.budget_min_new_coverage_fraction
        )
        lab_novel = bool(
            photometric_route
            and self.photometric_budget_admission_mode == "channel_or_lab"
            and evaluation.get("lab_novel_to_all_experts", False)
        )
        history_novel = bool(
            dict(candidate_evidence or {}).get("history_match", {}).get(
                "matched", False
            )
        )
        evaluation["channel_novel"] = channel_novel
        evaluation["lab_novel"] = lab_novel
        evaluation["history_novel"] = history_novel
        if history_novel:
            evaluation["candidate_admission_evidence"] = "history"
        elif channel_novel and lab_novel:
            evaluation["candidate_admission_evidence"] = "channel_and_lab"
        elif channel_novel:
            evaluation["candidate_admission_evidence"] = "channel"
        elif lab_novel:
            evaluation["candidate_admission_evidence"] = "lab"
        else:
            evaluation["candidate_admission_evidence"] = None

        if new_coverage_fraction is None and not lab_novel and not history_novel:
            evaluation["outcome"] = "missing_candidate_support"
            self.last_budget_replacement_evaluation = evaluation
            return None, None
        if not channel_novel and not lab_novel and not history_novel:
            evaluation["outcome"] = "insufficient_new_coverage"
            self.last_budget_replacement_evaluation = evaluation
            return None, None

        virtual_swap_profiles = None
        safe_virtual_victim_ids = None
        if self.budget_virtual_swap_enabled:
            task_new_coverage = bool(
                channel_novel or lab_novel or history_novel
            )
            evaluation["task_new_coverage"] = bool(task_new_coverage)
            virtual_swap_profiles = self._virtual_swap_profiles(
                candidate_members,
                support_threshold,
                candidate_evidence=candidate_evidence,
            )
            evaluation["virtual_swap_enabled"] = True
            evaluation["virtual_swap_profiles"] = virtual_swap_profiles
            if not virtual_swap_profiles:
                evaluation["outcome"] = "virtual_swap_insufficient_history"
                self.last_budget_replacement_evaluation = evaluation
                return None, None
            safe_virtual_victim_ids = {
                int(expert_id)
                for expert_id, profile in virtual_swap_profiles.items()
                if profile.get("safe", False)
                and profile.get("net_task_gain", 0.0) > 0.0
            }
            if not safe_virtual_victim_ids:
                has_safe_profile = any(
                    profile.get("safe", False)
                    for profile in virtual_swap_profiles.values()
                )
                evaluation["outcome"] = (
                    "virtual_swap_no_positive_task_gain"
                    if has_safe_profile
                    else "virtual_swap_rejected"
                )
                self.last_budget_replacement_evaluation = evaluation
                return None, None
        else:
            evaluation["virtual_swap_enabled"] = False

        coverage_profiles = self._expert_coverage_profiles()
        lab_uniqueness_profiles = self._expert_lab_uniqueness_profiles()
        if self.budget_coverage_loss_enabled:
            if self.budget_coverage_joint_eviction_enabled:
                joint_profiles = self._coverage_loss_joint_eviction_profiles(
                    candidate_members, support_threshold
                )
                active_by_id = {
                    int(expert["expert_id"]): expert
                    for expert in self.active_experts
                }
                ranked_entity_ids = sorted(
                    [
                        entity_id
                        for entity_id in joint_profiles
                        if entity_id == "candidate"
                        or safe_virtual_victim_ids is None
                        or int(entity_id) in safe_virtual_victim_ids
                    ],
                    key=lambda entity_id: (
                        joint_profiles[entity_id]["eviction_loss"],
                        entity_id != "candidate",
                        (
                            active_by_id[entity_id].get(
                                "lifecycle_state", "active"
                            )
                            != "dormant"
                            if entity_id != "candidate"
                            else False
                        ),
                        (
                            int(active_by_id[entity_id]["last_supported_step"])
                            if entity_id != "candidate"
                            else -1
                        ),
                        str(entity_id),
                    ),
                )
                evicted_entity_id = ranked_entity_ids[0]
                eviction_profile = joint_profiles[evicted_entity_id]
                evaluation.update(
                    {
                        "coverage_loss_enabled": True,
                        "coverage_selection_mode": "joint_minimum_loss",
                        "coverage_joint_eviction_profiles": joint_profiles,
                        "evicted_entity_id": evicted_entity_id,
                        "coverage_eviction_loss": float(
                            eviction_profile["eviction_loss"]
                        ),
                    }
                )
                if evicted_entity_id == "candidate":
                    evaluation["outcome"] = (
                        "candidate_evicted_minimum_coverage_loss"
                    )
                    self.last_budget_replacement_evaluation = evaluation
                    return None, None

                victim_id = int(evicted_entity_id)
                victim = active_by_id[victim_id]
                evaluation.update(
                    {
                        "victim_expert_id": victim_id,
                        "victim_coverage": dict(
                            coverage_profiles[victim_id]
                        ),
                        "victim_lab_uniqueness": dict(
                            lab_uniqueness_profiles[victim_id]
                        ),
                        "victim_lifecycle_state": victim.get(
                            "lifecycle_state", "active"
                        ),
                        "outcome": "accepted",
                    }
                )
                self.last_budget_replacement_evaluation = evaluation
                return victim_id, {
                    "admission_route": admission_route,
                    "candidate_admission_evidence": evaluation[
                        "candidate_admission_evidence"
                    ],
                    "replacement_reason": (
                        "coreset_joint_minimum_loss_replacement"
                    ),
                    "new_coverage_fraction": (
                        float(new_coverage_fraction)
                        if new_coverage_fraction is not None
                        else None
                    ),
                    "new_coverage_count": int(new_coverage_count),
                    "candidate_observation_count": int(
                        candidate_observation_count
                    ),
                    "victim_coverage": dict(coverage_profiles[victim_id]),
                    "victim_lab_uniqueness": dict(
                        lab_uniqueness_profiles[victim_id]
                    ),
                    "victim_lifecycle_state": victim.get(
                        "lifecycle_state", "active"
                    ),
                    "coverage_eviction_loss": float(
                        eviction_profile["eviction_loss"]
                    ),
                    "coverage_loss_profile": dict(eviction_profile),
                }

            coverage_loss_profiles = (
                self._coverage_loss_replacement_profiles(
                    candidate_members, support_threshold
                )
            )
            replacement_candidates = [
                expert
                for expert in self.active_experts
                if safe_virtual_victim_ids is None
                or int(expert["expert_id"]) in safe_virtual_victim_ids
            ]
            if not replacement_candidates:
                evaluation["outcome"] = "virtual_swap_rejected"
                self.last_budget_replacement_evaluation = evaluation
                return None, None
            if self.budget_coverage_min_loss_only:
                ranked = sorted(
                    replacement_candidates,
                    key=lambda expert: (
                        coverage_loss_profiles[int(expert["expert_id"])][
                            "removal_loss"
                        ],
                        expert.get("lifecycle_state", "active")
                        != "dormant",
                        int(expert["last_supported_step"]),
                        int(expert["expert_id"]),
                    ),
                )
            else:
                ranked = sorted(
                    replacement_candidates,
                    key=lambda expert: (
                        -coverage_loss_profiles[int(expert["expert_id"])][
                            "swap_gain"
                        ],
                        coverage_loss_profiles[int(expert["expert_id"])][
                            "removal_loss"
                        ],
                        expert.get("lifecycle_state", "active")
                        != "dormant",
                        int(expert["last_supported_step"]),
                        int(expert["expert_id"]),
                    ),
                )
            victim = ranked[0]
            victim_id = int(victim["expert_id"])
            victim_loss = coverage_loss_profiles[victim_id]
            evaluation.update(
                {
                    "coverage_loss_enabled": True,
                    "coverage_selection_mode": (
                        "minimum_removal_loss"
                        if self.budget_coverage_min_loss_only
                        else "positive_swap_gain"
                    ),
                    "coverage_loss_profiles": coverage_loss_profiles,
                    "victim_expert_id": victim_id,
                    "victim_coverage": dict(coverage_profiles[victim_id]),
                    "victim_lab_uniqueness": dict(
                        lab_uniqueness_profiles[victim_id]
                    ),
                    "victim_lifecycle_state": victim.get(
                        "lifecycle_state", "active"
                    ),
                    "coverage_removal_loss": float(
                        victim_loss["removal_loss"]
                    ),
                    "coverage_swap_gain": float(victim_loss["swap_gain"]),
                }
            )
            if (
                not self.budget_coverage_min_loss_only
                and
                victim_loss["swap_gain"]
                <= self.budget_replacement_margin
            ):
                evaluation["outcome"] = "no_positive_coverage_gain"
                self.last_budget_replacement_evaluation = evaluation
                return None, None
            evaluation["outcome"] = "accepted"
            self.last_budget_replacement_evaluation = evaluation
            return victim_id, {
                "admission_route": admission_route,
                "candidate_admission_evidence": evaluation[
                    "candidate_admission_evidence"
                ],
                "replacement_reason": (
                    "coreset_minimum_coverage_loss_replacement"
                    if self.budget_coverage_min_loss_only
                    else "coreset_coverage_replacement"
                ),
                "new_coverage_fraction": (
                    float(new_coverage_fraction)
                    if new_coverage_fraction is not None
                    else None
                ),
                "new_coverage_count": int(new_coverage_count),
                "candidate_observation_count": int(
                    candidate_observation_count
                ),
                "victim_coverage": dict(coverage_profiles[victim_id]),
                "victim_lab_uniqueness": dict(
                    lab_uniqueness_profiles[victim_id]
                ),
                "victim_lifecycle_state": victim.get(
                    "lifecycle_state", "active"
                ),
                "coverage_removal_loss": float(
                    victim_loss["removal_loss"]
                ),
                "coverage_swap_gain": float(victim_loss["swap_gain"]),
                "coverage_loss_profile": dict(victim_loss),
            }
        reference_step = max(
            [int(member.get("step", self.last_step)) for member in candidate_members]
            or [self.last_step]
        )
        expert_contributions = self._expert_contribution_profiles(
            coverage_profiles,
            lab_uniqueness_profiles,
            reference_step,
        )
        candidate_contribution = self._candidate_contribution_profile(
            evaluation,
            candidate_evidence,
        )
        eligible = []
        for expert in self.active_experts:
            expert_id = int(expert["expert_id"])
            profile = coverage_profiles[expert_id]
            lab_profile = lab_uniqueness_profiles[expert_id]
            expert_contribution = expert_contributions[expert_id]
            replaceable = bool(
                self.budget_virtual_swap_enabled
                or candidate_contribution["contribution_score"]
                > expert_contribution["contribution_score"]
                + self.budget_replacement_margin
            )
            if not replaceable:
                continue
            eligible.append(
                (expert, profile, lab_profile, expert_contribution)
            )
        if not eligible:
            evaluation["outcome"] = "no_redundant_victim"
            evaluation["expert_coverage_profiles"] = coverage_profiles
            evaluation["expert_lab_uniqueness_profiles"] = (
                lab_uniqueness_profiles
            )
            evaluation["expert_contribution_profiles"] = expert_contributions
            evaluation["candidate_contribution"] = candidate_contribution
            self.last_budget_replacement_evaluation = evaluation
            return None, None

        if safe_virtual_victim_ids is not None:
            eligible = [
                item
                for item in eligible
                if int(item[0]["expert_id"]) in safe_virtual_victim_ids
            ]
            if not eligible:
                evaluation["outcome"] = "virtual_swap_rejected"
                self.last_budget_replacement_evaluation = evaluation
                return None, None

        lab_aware_replacement = bool(
            photometric_route
            and self.photometric_budget_admission_mode == "channel_or_lab"
        )
        patch_victim_profile = None
        semantic_replacement_losses = None
        use_patch_victim_selection = bool(
            self.budget_patch_victim_selection_enabled
            and (channel_novel or history_novel)
        )
        if use_patch_victim_selection:
            semantic_replacement_losses = {
                int(item[0]["expert_id"]): {
                    "channel_loss": max(
                        0.0,
                        item[3]["channel_unique_score"]
                        - candidate_contribution[
                            "channel_new_coverage_score"
                        ],
                    ),
                    "lab_loss": max(
                        0.0,
                        item[3]["lab_unique_score"]
                        - candidate_contribution["lab_unique_score"],
                    ),
                }
                for item in eligible
            }
            for profile in semantic_replacement_losses.values():
                profile["total_loss"] = (
                    profile["channel_loss"] + profile["lab_loss"]
                )
            if self.budget_patch_kcenter_victim_selection_enabled:
                joint_profiles = self._patch_kcenter_eviction_profiles(
                    candidate_members, support_threshold
                )
            else:
                joint_profiles = self._coverage_loss_joint_eviction_profiles(
                    candidate_members, support_threshold
                )
            (
                victim,
                victim_profile,
                victim_lab_profile,
                victim_contribution,
            ) = min(
                eligible,
                key=lambda item: (
                    (
                        virtual_swap_profiles[
                            int(item[0]["expert_id"])
                        ]["net_task_gain"]
                        * -1.0
                        if virtual_swap_profiles is not None
                        else 0.0
                    ),
                    semantic_replacement_losses[
                        int(item[0]["expert_id"])
                    ]["total_loss"],
                    joint_profiles[int(item[0]["expert_id"])][
                        "eviction_loss"
                    ],
                    item[0].get("lifecycle_state", "active")
                    != "dormant",
                    int(item[0]["last_supported_step"]),
                    int(item[0]["expert_id"]),
                ),
            )
            patch_victim_profile = joint_profiles[
                int(victim["expert_id"])
            ]
        elif lab_aware_replacement:
            victim, victim_profile, victim_lab_profile, victim_contribution = min(
                eligible,
                key=lambda item: (
                    item[0].get("lifecycle_state", "active") != "dormant",
                    item[3]["contribution_score"],
                    int(item[0]["support_count"]),
                    int(item[0]["last_supported_step"]),
                    int(item[0]["admission_step"]),
                ),
            )
        else:
            victim, victim_profile, victim_lab_profile, victim_contribution = min(
                eligible,
                key=lambda item: (
                    item[0].get("lifecycle_state", "active") != "dormant",
                    item[3]["contribution_score"],
                    int(item[0]["support_count"]),
                    int(item[0]["last_supported_step"]),
                    int(item[0]["admission_step"]),
                ),
            )
        evaluation.update(
            {
                "outcome": "accepted",
                "victim_expert_id": int(victim["expert_id"]),
                "victim_coverage": dict(victim_profile),
                "victim_lab_uniqueness": dict(victim_lab_profile),
                "victim_contribution": dict(victim_contribution),
                "candidate_contribution": dict(candidate_contribution),
                "victim_lifecycle_state": victim.get(
                    "lifecycle_state", "active"
                ),
                "lab_aware_replacement": lab_aware_replacement,
                "expert_coverage_profiles": coverage_profiles,
                "expert_lab_uniqueness_profiles": lab_uniqueness_profiles,
                "expert_contribution_profiles": expert_contributions,
                "semantic_replacement_losses": (
                    semantic_replacement_losses
                    if use_patch_victim_selection
                    else None
                ),
                "semantic_selection_mode": (
                    "minimum_uncovered_channel_lab_loss"
                    if use_patch_victim_selection
                    else None
                ),
                "coverage_selection_mode": (
                    "patch_kcenter_minimum_victim_loss"
                    if (
                        self.budget_patch_kcenter_victim_selection_enabled
                        and use_patch_victim_selection
                    )
                    else "patch_minimum_victim_loss"
                    if use_patch_victim_selection
                    else None
                ),
                "coverage_joint_eviction_profiles": (
                    joint_profiles
                    if use_patch_victim_selection
                    else None
                ),
                "coverage_eviction_loss": (
                    float(patch_victim_profile["eviction_loss"])
                    if patch_victim_profile is not None
                    else None
                ),
                "virtual_swap_selected_victim": (
                    virtual_swap_profiles.get(int(victim["expert_id"]))
                    if virtual_swap_profiles is not None
                    else None
                ),
            }
        )
        self.last_budget_replacement_evaluation = evaluation
        return int(victim["expert_id"]), {
            "admission_route": admission_route,
            "candidate_admission_evidence": evaluation[
                "candidate_admission_evidence"
            ],
            "replacement_reason": (
                "history_reentry_swap"
                if history_novel
                else "coreset_patch_victim_replacement"
                if use_patch_victim_selection
                else "lab_or_coverage_budget_replacement"
                if lab_novel
                else "coverage_budget_replacement"
            ),
            "new_coverage_fraction": (
                float(new_coverage_fraction)
                if new_coverage_fraction is not None
                else None
            ),
            "new_coverage_count": int(new_coverage_count),
            "candidate_observation_count": int(candidate_observation_count),
            "victim_coverage": dict(victim_profile),
            "victim_lab_uniqueness": dict(victim_lab_profile),
            "victim_contribution": dict(victim_contribution),
            "candidate_contribution": dict(candidate_contribution),
            "victim_lifecycle_state": victim.get(
                "lifecycle_state", "active"
            ),
            "coverage_eviction_loss": (
                float(patch_victim_profile["eviction_loss"])
                if patch_victim_profile is not None
                else None
            ),
            "coverage_loss_profile": (
                dict(patch_victim_profile)
                if patch_victim_profile is not None
                else None
            ),
        }

    def _replace_budgeted_expert(self, expert_id, step, audit):
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
        replacement_reason = dict(audit or {}).get(
            "replacement_reason", "coverage_budget_replacement"
        )
        if replacement_reason == "coverage_budget_replacement":
            expert["last_event"] = "replaced_coverage_budget"
        elif replacement_reason in {
            "coreset_coverage_replacement",
            "coreset_minimum_coverage_loss_replacement",
            "coreset_joint_minimum_loss_replacement",
            "coreset_patch_victim_replacement",
        }:
            expert["last_event"] = "replaced_coreset_coverage"
        elif replacement_reason == "history_reentry_swap":
            expert["last_event"] = "replaced_history_reentry"
        else:
            expert["last_event"] = "replaced_lab_or_coverage_budget"
        expert["lifecycle_state"] = "retired"
        cluster = self.clusters[expert["cluster_id"]]
        if cluster["active_expert_id"] == expert["expert_id"]:
            cluster["active_expert_id"] = None
        self.retired_experts.append(expert)
        self._register_expert_historical_state(
            expert,
            "retired",
            step,
        )
        self.last_budget_replacement = {
            "expert_id": int(expert_id),
            "step": int(step),
            "reason": replacement_reason,
            **dict(audit or {}),
        }
        return True

    def _admit_quarantined_member(
        self,
        member,
        step,
        cluster_members=None,
        admission_route="new_distribution",
        support_threshold=None,
        candidate_evidence=None,
    ):
        cluster_members = list(cluster_members or [member])
        self._refresh_candidate_reliabilities(cluster_members)
        if len(self.active_experts) >= self.committee_cap:
            replacement_expert_id, replacement_audit = (
                self._select_budget_replacement(
                    cluster_members,
                    support_threshold,
                    admission_route,
                    candidate_evidence=candidate_evidence,
                )
            )
            if (
                replacement_expert_id is None
                or not self._replace_budgeted_expert(
                    replacement_expert_id,
                    step,
                    replacement_audit,
                )
            ):
                return None, None

        cluster_id = len(self.clusters)
        self.clusters.append(
            {
                "embedding_sum": self._candidate_weighted_embedding_sum(
                    cluster_members
                ),
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
            "lifecycle_state": "active",
            "last_channel_support": None,
            "representative_mode": self.representative_mode,
            "admission_ttl_mode": self.admission_ttl_mode,
            "admission_route": admission_route,
            "lab_state": (
                self._build_lab_state(cluster_members, step=step)
                if admission_route.startswith("photometric_drift")
                else None
            ),
            "admission_candidate_ms_score": member["ms_score"],
            "admission_candidate_channel_support": member[
                "channel_support"
            ],
        }
        if self.candidate_reliability_enabled:
            expert["member_reliability"] = float(
                member.get("member_reliability", 1.0)
            )
            expert["scoring_patch_reliabilities"] = [
                float(value)
                for value in torch.as_tensor(
                    member.get(
                        "scoring_patch_reliabilities",
                        member.get("patch_reliabilities"),
                    ),
                    dtype=torch.float32,
                ).tolist()
            ]
            expert["reference_reliability_penalty"] = float(
                self.reference_reliability_penalty
            )
            expert["reliability_audit"] = dict(
                member.get("reliability_audit", {})
            )
            expert["reference_member_image_ids"] = [
                int(item["image_id"]) for item in cluster_members
            ]
            expert["reference_member_reliabilities"] = [
                float(item.get("member_reliability", 1.0))
                for item in cluster_members
            ]
        self.active_experts.append(expert)
        self._register_expert_historical_state(
            expert,
            "active",
            step,
            state_id=(
                candidate_evidence.get("associated_state_id")
                if candidate_evidence is not None
                else None
            ),
        )
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
        current_expert_supports=None,
        support_threshold=None,
        patch_reliabilities=None,
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
            memory_patch_ids=memory_patch_ids,
            patch_reliabilities=patch_reliabilities,
            lab_feature=lab_feature,
            candidate_route=(
                trigger_route
                or self.novel_quarantine_route
                or "new_distribution"
            ),
            historical_ms_threshold=historical_ms_threshold,
            current_expert_supports=current_expert_supports,
        )
        fixed_scale_photometric = bool(
            (
                trigger_route == "photometric_drift"
                or self.novel_quarantine_route == "photometric_drift"
            )
            and self.photometric_candidate_window_mode != "history"
        )
        if (
            fixed_scale_photometric
            and not self.photometric_adaptive_candidate_evidence
            and not member["historical_low_risk"]
        ):
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
            self._refresh_candidate_reliabilities(self.novel_quarantine)
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
            self._refresh_candidate_reliabilities(self.novel_quarantine)
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
        self._refresh_candidate_reliabilities(self.novel_quarantine)
        photometric_required_count = self.photometric_min_low_risk_members
        provisional_profile = None
        history_reentry_match = None
        adaptive_admission_evidence = None
        state_aware_photometric = bool(
            self.novel_quarantine_route == "photometric_drift"
            and self.photometric_candidate_window_mode
            == "dynamic_low_risk"
            and (
                self.photometric_state_distribution_enabled
                or self.photometric_temporary_state_enabled
                or self.photometric_adaptive_candidate_evidence
            )
        )
        adaptive_photometric = bool(
            state_aware_photometric
            and self.photometric_adaptive_candidate_evidence
        )
        if state_aware_photometric:
            eligible_members = [
                item
                for item in self.novel_quarantine
                if (
                    item.get("patch_reliabilities") is not None
                    if self.candidate_reliability_enabled
                    else bool(item.get("memory_patch_ids"))
                )
            ]
            if len(eligible_members) >= 2:
                provisional_profile = self._photometric_pool_profile(
                    eligible_members,
                    self.photometric_trigger_expert_id,
                    self.photometric_trigger_lab_center,
                    required_count=2,
                )
                adaptive_admission_evidence = (
                    self._candidate_admission_evidence(
                        eligible_members,
                        provisional_profile,
                        support_threshold,
                    )
                )
                decision = adaptive_admission_evidence["decision"]
                if decision == "history_reentry":
                    photometric_required_count = 2
                    history_reentry_match = adaptive_admission_evidence[
                        "history_match"
                    ]
                    provisional_profile["history_reentry"] = dict(
                        history_reentry_match
                    )
                elif decision == "new_scoring_expert":
                    photometric_required_count = 2
                elif decision == "new_temporary_state":
                    selected_members = adaptive_admission_evidence[
                        "selected_members"
                    ]
                    temporary_support_evidence = (
                        adaptive_admission_evidence.get(
                            "temporary_support_evidence"
                        )
                        if self.photometric_denstream_support_enabled
                        else None
                    )
                    initial_state_status = (
                        "stable"
                        if temporary_support_evidence is not None
                        and temporary_support_evidence.get("increment", 0.0)
                        >= self.photometric_denstream_mature_support
                        else "temporary"
                    )
                    recorded_state = self._record_photometric_state(
                        selected_members,
                        step,
                        state_status=initial_state_status,
                        source_type=(
                            "photometric_only"
                            if initial_state_status == "stable"
                            else "temporary"
                        ),
                        temporary_support_evidence=(
                            temporary_support_evidence
                        ),
                    )
                    pool = list(self.novel_quarantine)
                    self.novel_quarantine = None
                    self.novel_quarantine_route = None
                    self.photometric_trigger_expert_id = None
                    self.photometric_trigger_lab_center = None
                    self.photometric_support_streak = 0
                    self.last_novel_quarantine_event = {
                        "decision": (
                            "recorded_photometric_state_only"
                            if initial_state_status == "stable"
                            else "recorded_temporary_photometric_state"
                        ),
                        "confirmation_reason": (
                            "denstream_effective_support_matured"
                            if initial_state_status == "stable"
                            else "first_consistent_state_event"
                        ),
                        "start_step": int(pool[0]["step"]),
                        "end_step": int(step),
                        "size": len(pool),
                        "route": "photometric_drift",
                        "created_full_expert": False,
                        "historical_state": recorded_state,
                        "photometric_profile": provisional_profile,
                        "candidate_evidence": {
                            key: value
                            for key, value in adaptive_admission_evidence.items()
                            if key != "selected_members"
                        },
                        "selected_image_ids": [
                            int(item["image_id"])
                            for item in selected_members
                        ],
                    }
                    if initial_state_status == "stable":
                        self._rebase_photometric_trigger_detector(
                            selected_members,
                            step,
                        )
                    self._finish_photometric_episode(
                        provisional_profile.get("selected_lab_center")
                        or self._photometric_pool_lab_center(
                            selected_members
                        )
                    )
                    return None, None
                elif decision in {
                    "update_temporary_state",
                    "update_existing_state",
                    "existing_state_supported",
                }:
                    selected_members = adaptive_admission_evidence[
                        "selected_members"
                    ]
                    state_update = (
                        self._update_matched_historical_state(
                            adaptive_admission_evidence["history_match"],
                            selected_members,
                            step,
                            temporary_support_evidence=(
                                adaptive_admission_evidence.get(
                                    "temporary_support_evidence"
                                )
                                if self.photometric_denstream_support_enabled
                                else None
                            ),
                        )
                        if decision != "existing_state_supported"
                        else adaptive_admission_evidence["history_match"].get(
                            "matched_profile"
                        )
                    )
                    state_is_stable = bool(
                        state_update is not None
                        and state_update.get("state_status_after") == "stable"
                    )
                    should_materialize = bool(
                        decision == "update_temporary_state"
                        and state_is_stable
                        and adaptive_admission_evidence[
                            "has_reliable_patch_memory"
                        ]
                        and adaptive_admission_evidence["channel_novel"]
                    )
                    if should_materialize:
                        adaptive_admission_evidence[
                            "decision"
                        ] = "new_scoring_expert"
                        adaptive_admission_evidence[
                            "associated_state_id"
                        ] = adaptive_admission_evidence[
                            "history_match"
                        ]["matched_state_id"]
                        photometric_required_count = 2
                    else:
                        pool = list(self.novel_quarantine)
                        self.novel_quarantine = None
                        self.novel_quarantine_route = None
                        self.photometric_trigger_expert_id = None
                        self.photometric_trigger_lab_center = None
                        self.photometric_support_streak = 0
                        event_decision = (
                            "existing_photometric_state_supported"
                            if decision == "existing_state_supported"
                            else
                            "updated_existing_photometric_state"
                            if decision == "update_existing_state"
                            else "matured_photometric_state_only"
                            if state_is_stable
                            else "updated_temporary_photometric_state"
                        )
                        self.last_novel_quarantine_event = {
                            "decision": event_decision,
                            "confirmation_reason": decision,
                            "start_step": int(pool[0]["step"]),
                            "end_step": int(step),
                            "size": len(pool),
                            "route": "photometric_drift",
                            "created_full_expert": False,
                            "historical_state": state_update,
                            "photometric_profile": provisional_profile,
                            "candidate_evidence": {
                                key: value
                                for key, value in adaptive_admission_evidence.items()
                                if key != "selected_members"
                            },
                            "selected_image_ids": [
                                int(item["image_id"])
                                for item in selected_members
                            ],
                        }
                        if (
                            state_is_stable
                            and state_update is not None
                            and state_update.get("active_expert_id") is None
                        ):
                            self._rebase_photometric_trigger_detector(
                                selected_members,
                                step,
                            )
                        self._finish_photometric_episode(
                            provisional_profile.get("selected_lab_center")
                            or self._photometric_pool_lab_center(
                                selected_members
                            )
                        )
                        return None, None
        if (
            not state_aware_photometric
            and
            self.novel_quarantine_route == "photometric_drift"
            and self.photometric_candidate_window_mode == "dynamic_low_risk"
            and self.photometric_history_reentry_enabled
        ):
            early_low_risk_members = [
                item
                for item in self.novel_quarantine
                if item.get("historical_low_risk", False)
            ]
            if (
                len(early_low_risk_members)
                >= self.photometric_history_reentry_members
            ):
                history_profile = self._photometric_pool_profile(
                    early_low_risk_members,
                    self.photometric_trigger_expert_id,
                    self.photometric_trigger_lab_center,
                    required_count=(
                        self.photometric_history_reentry_members
                    ),
                )
                if history_profile["consistent"]:
                    selected_ids = set(
                        int(image_id)
                        for image_id in history_profile.get(
                            "selected_inlier_image_ids", []
                        )
                    )
                    selected_members = [
                        item
                        for item in early_low_risk_members
                        if int(item["image_id"]) in selected_ids
                    ]
                    history_reentry_match = (
                        self._retired_photometric_state_match(
                            selected_members
                        )
                    )
                    if history_reentry_match["matched"]:
                        photometric_required_count = (
                            self.photometric_history_reentry_members
                        )
                        history_profile["history_reentry"] = dict(
                            history_reentry_match
                        )
                        provisional_profile = history_profile
        if self.novel_quarantine_route == "photometric_drift":
            if self.photometric_candidate_window_mode == "fixed_low_risk":
                required_steps = self.photometric_fixed_low_risk_members
            elif (
                self.photometric_candidate_window_mode
                == "dynamic_low_risk"
            ):
                required_steps = (
                    2
                    if adaptive_photometric
                    else self.photometric_dynamic_min_low_risk_members
                )
                if (
                    not adaptive_photometric
                    and self.photometric_history_reentry_enabled
                ):
                    required_steps = min(
                        required_steps,
                        self.photometric_history_reentry_members,
                    )
            elif self.photometric_fitness_history_enabled:
                effective_size = float(
                    (self.last_fitness_history_profile or {}).get(
                        "effective_history_size", 0.0
                    )
                )
                required_steps = max(2, int(math.ceil(effective_size)) + 1)
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
                "required_confirmed_members": photometric_required_count,
                "route": self.novel_quarantine_route,
            }
            return None, None

        if (
            self.novel_quarantine_route == "photometric_drift"
            and self.photometric_candidate_window_mode
            == "dynamic_low_risk"
        ):
            if adaptive_photometric:
                decision = (
                    adaptive_admission_evidence.get("decision")
                    if adaptive_admission_evidence is not None
                    else "keep_collecting"
                )
                if (
                    decision == "keep_collecting"
                    and len(self.novel_quarantine)
                    < self.photometric_dynamic_max_low_risk_members
                ):
                    self.last_novel_quarantine_event = {
                        "decision": "collecting_adaptive_evidence",
                        "start_step": int(
                            self.novel_quarantine[0]["step"]
                        ),
                        "size": len(self.novel_quarantine),
                        "required_steps": (
                            self.photometric_dynamic_max_low_risk_members
                        ),
                        "required_confirmed_members": 2,
                        "route": self.novel_quarantine_route,
                        "photometric_profile": provisional_profile,
                        "candidate_evidence": (
                            {
                                key: value
                                for key, value in (
                                    adaptive_admission_evidence or {}
                                ).items()
                                if key != "selected_members"
                            }
                        ),
                    }
                    return None, None
            elif provisional_profile is None:
                provisional_low_risk_members = [
                    item
                    for item in self.novel_quarantine
                    if item.get("historical_low_risk", False)
                ]
                if (
                    len(provisional_low_risk_members)
                    >= photometric_required_count
                ):
                    provisional_profile = self._photometric_pool_profile(
                        provisional_low_risk_members,
                        self.photometric_trigger_expert_id,
                        self.photometric_trigger_lab_center,
                        required_count=photometric_required_count,
                    )
                else:
                    provisional_profile = {
                        "consistent": False,
                        "reason": "insufficient_low_risk",
                        "evaluated_member_count": len(
                            provisional_low_risk_members
                        ),
                        "required_count": photometric_required_count,
                    }
            if (
                not adaptive_photometric
                and
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
                    "required_confirmed_members": (
                        photometric_required_count
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
        photometric_admission_members = None
        if pool_route == "photometric_drift":
            photometric_low_risk_members = [
                member
                for member in pool
                if member.get("historical_low_risk", False)
            ]
            if adaptive_photometric:
                if (
                    adaptive_admission_evidence is None
                    or adaptive_admission_evidence["decision"]
                    == "keep_collecting"
                ):
                    self.photometric_support_streak = 0
                    self.last_novel_quarantine_event = {
                        "decision": "rejected_adaptive_evidence_timeout",
                        "confirmation_reason": "insufficient_joint_evidence",
                        "start_step": int(pool[0]["step"]),
                        "end_step": int(step),
                        "size": len(pool),
                        "route": pool_route,
                        "created_full_expert": False,
                        "photometric_profile": provisional_profile,
                        "candidate_evidence": (
                            {
                                key: value
                                for key, value in (
                                    adaptive_admission_evidence or {}
                                ).items()
                                if key != "selected_members"
                            }
                        ),
                    }
                    self._finish_photometric_episode(
                        self._photometric_pool_lab_center(pool)
                    )
                    return None, None
                photometric_profile = provisional_profile
                photometric_admission_members = (
                    adaptive_admission_evidence["selected_members"]
                )
            elif (
                len(photometric_low_risk_members)
                < photometric_required_count
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
                        photometric_required_count
                    ),
                    "low_risk_image_ids": [
                        int(member["image_id"])
                        for member in photometric_low_risk_members
                    ],
                }
                self._finish_photometric_episode(
                    self._photometric_pool_lab_center(
                        photometric_low_risk_members or pool
                    )
                )
                return None, None
            if adaptive_photometric:
                pass
            elif provisional_profile is not None and provisional_profile.get(
                "history_reentry", {}
            ).get("matched", False):
                photometric_profile = provisional_profile
            else:
                photometric_profile = self._photometric_pool_profile(
                    photometric_low_risk_members,
                    pool_trigger_expert_id,
                    pool_trigger_lab_center,
                    required_count=photometric_required_count,
                )
            if (
                not adaptive_photometric
                and not photometric_profile["consistent"]
            ):
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
                        photometric_required_count
                    ),
                    "low_risk_image_ids": [
                        int(member["image_id"])
                        for member in photometric_low_risk_members
                    ],
                }
                self._finish_photometric_episode(
                    photometric_profile.get("selected_lab_center")
                    or self._photometric_pool_lab_center(
                        photometric_low_risk_members
                    )
                )
                return None, None
            if not adaptive_photometric:
                selected_image_ids = set(
                    int(image_id)
                    for image_id in photometric_profile.get(
                        "selected_inlier_image_ids",
                        [
                            int(member["image_id"])
                            for member in photometric_low_risk_members
                        ],
                    )
                )
                photometric_admission_members = [
                    member
                    for member in photometric_low_risk_members
                    if int(member["image_id"]) in selected_image_ids
                ]

        admission_pool = (
            photometric_admission_members
            if photometric_admission_members is not None
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
        photometric_history_reentry = bool(
            pool_route == "photometric_drift"
            and photometric_profile is not None
            and photometric_profile.get("history_reentry", {}).get(
                "matched", False
            )
        )
        admission_route_base = (
            "photometric_drift_history_reentry"
            if photometric_history_reentry
            else "photometric_drift_adaptive_coverage"
            if (
                adaptive_photometric
                and adaptive_admission_evidence is not None
                and adaptive_admission_evidence["decision"]
                == "new_scoring_expert"
            )
            else pool_route
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
                    admission_pool
                    if pool_route == "photometric_drift"
                    else pool,
                    min_keep=(
                        photometric_required_count
                        if (
                            pool_route == "photometric_drift"
                            and self.photometric_candidate_window_mode
                            != "history"
                        )
                        else 1
                    ),
                )
                if (
                    pool_route == "photometric_drift"
                    and self.photometric_candidate_window_mode != "history"
                    and len(representatives)
                    < photometric_required_count
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
                        photometric_required_count
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
                    self._finish_photometric_episode(
                        photometric_profile.get("selected_lab_center")
                        if photometric_profile is not None
                        else self._photometric_pool_lab_center(admission_pool)
                    )
                return None, None
            if self.novel_admission_mode == "channel_gate":
                representative = representatives[0]
                admission_route = f"{admission_route_base}_channel_gate"
                if photometric_history_reentry:
                    cluster_id, expert_id = self._activate_historical_state(
                        history_reentry_match,
                        representatives,
                        step,
                        support_threshold,
                        admission_route,
                        candidate_evidence=adaptive_admission_evidence,
                    )
                else:
                    cluster_id, expert_id = self._admit_quarantined_member(
                        representative,
                        step,
                        cluster_members=(
                            representatives
                            if pool_route == "photometric_drift"
                            else [representative]
                        ),
                        admission_route=admission_route,
                        support_threshold=support_threshold,
                        candidate_evidence=adaptive_admission_evidence,
                    )
                retained_state = None
                replacement_outcome = (
                    (self.last_budget_replacement_evaluation or {}).get(
                        "outcome"
                    )
                )
                if (
                    not photometric_history_reentry
                    and expert_id is None
                    and replacement_outcome in {
                        "virtual_swap_rejected",
                        "virtual_swap_insufficient_history",
                        "virtual_swap_no_positive_task_gain",
                    }
                ):
                    retained_state = self._record_photometric_state(
                        representatives,
                        step,
                        state_status="stable",
                        source_type="photometric_only",
                    )
                history_decision = (
                    self.last_history_reentry_event.get("decision")
                    if photometric_history_reentry
                    and self.last_history_reentry_event is not None
                    else None
                )
                self.last_novel_quarantine_event = {
                    "decision": (
                        history_decision
                        if history_decision is not None
                        else "confirmed_channel_gate"
                        if expert_id is not None
                        else "retained_history_no_safe_swap"
                        if retained_state is not None
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
                    "confirmation_reason": (
                        adaptive_admission_evidence["decision"]
                        if adaptive_admission_evidence is not None
                        else None
                    ),
                    "created_full_expert": expert_id is not None,
                    "historical_state": retained_state,
                    "history_reentry": self.last_history_reentry_event,
                    "candidate_evidence": (
                        {
                            key: value
                            for key, value in adaptive_admission_evidence.items()
                            if key != "selected_members"
                        }
                        if adaptive_admission_evidence is not None
                        else None
                    ),
                    "photometric_profile": photometric_profile,
                    "low_risk_count": (
                        len(photometric_low_risk_members)
                        if photometric_low_risk_members is not None
                        else None
                    ),
                    "required_low_risk_count": (
                        photometric_required_count
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
                    self._finish_photometric_episode(
                        photometric_profile.get("selected_lab_center")
                        if photometric_profile is not None
                        else self._photometric_pool_lab_center(admission_pool)
                    )
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
                    photometric_required_count
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
                self._finish_photometric_episode(
                    photometric_profile.get("selected_lab_center")
                    if photometric_profile is not None
                    else self._photometric_pool_lab_center(admission_pool)
                )
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
                f"{admission_route_base}_provisional"
                if self.novel_admission_mode == "provisional"
                else f"{admission_route_base}_quarantine"
            ),
            support_threshold=support_threshold,
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
                photometric_required_count
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
            self._finish_photometric_episode(
                photometric_profile.get("selected_lab_center")
                if photometric_profile is not None
                else self._photometric_pool_lab_center(admission_pool)
            )
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
        patch_reliabilities=None,
    ):
        current_expert_supports = current_expert_supports or {}
        self.last_fitness_history_profile = None
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
            memory_patch_ids=memory_patch_ids,
            patch_reliabilities=patch_reliabilities,
            lab_feature=lab_feature,
            candidate_route=(candidate_route or "history"),
            historical_ms_threshold=ms_threshold,
            current_expert_supports=current_expert_supports,
        )
        if self.photometric_pool_abort_reason is not None:
            self._abort_photometric_quarantine(
                step,
                self.photometric_pool_abort_reason,
            )
            self.photometric_prebuffer.append(current_member)
            prebuffer_size = (
                self.photometric_history_max
                if (
                    self.photometric_relevant_history_enabled
                    or self.photometric_fitness_history_enabled
                )
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
            if self.photometric_fitness_history_enabled:
                (
                    photometric_seed_members,
                    self.last_fitness_history_profile,
                ) = self._fitness_history_profile(
                    self.photometric_prebuffer,
                    current_member,
                )
            elif self.photometric_relevant_history_enabled:
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
                memory_patch_ids=memory_patch_ids,
                patch_reliabilities=patch_reliabilities,
                lab_feature=lab_feature,
                photometric_seed_members=photometric_seed_members,
                photometric_trigger_expert_id=(
                    photometric_trigger_expert_id
                ),
                photometric_trigger_lab_center=(
                    photometric_trigger_lab_center
                ),
                historical_ms_threshold=ms_threshold,
                current_expert_supports=current_expert_supports,
                support_threshold=support_threshold,
            )
        )
        if (
            self.last_fitness_history_profile is not None
            and self.last_novel_quarantine_event is not None
        ):
            self.last_novel_quarantine_event["fitness_history"] = dict(
                self.last_fitness_history_profile
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
            if (
                self.photometric_relevant_history_enabled
                or self.photometric_fitness_history_enabled
            )
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

        cluster_member = {
                "step": step,
                "image_id": image_id,
                "embedding": embedding,
                "patch_prototypes": self._patch_coreset(
                    dino_patch_features,
                    patch_ids=(
                        None
                        if self.candidate_reliability_enabled
                        else memory_patch_ids
                    ),
                    patch_reliabilities=(
                        patch_reliabilities
                        if self.candidate_reliability_enabled
                        else None
                    ),
                ),
                "quality": float(quality),
                "ms_score": ms_score,
                "channel_support":
                    channel_support,
                "ms_threshold": ms_threshold,
                "support_threshold":
                    support_threshold,
                "candidate_route": candidate_route,
                "provisional_channel_support": provisional_channel_support,
                "current_expert_supports": self._normalized_expert_supports(
                    current_expert_supports
                ),
            }
        if self.candidate_reliability_enabled:
            base_reliabilities = (
                torch.as_tensor(
                    patch_reliabilities,
                    dtype=torch.float32,
                ).detach().cpu()
                if patch_reliabilities is not None
                else torch.ones(
                    dino_patch_features.shape[0],
                    dtype=torch.float32,
                )
            )
            cluster_member["base_patch_reliabilities"] = base_reliabilities
            cluster_member["patch_reliabilities"] = base_reliabilities.clone()
            cluster_member["reliability_patch_bank"] = (
                self._candidate_patch_bank(
                    dino_patch_features,
                    base_reliabilities,
                )
            )
            cluster_member["member_reliability"] = 0.5
        cluster["members"].append(cluster_member)
        # 更新 cluster 中心
        cluster["embedding_sum"] = self._candidate_weighted_embedding_sum(
            cluster["members"]
        )
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
            replacement_expert_id, replacement_audit = (
                self._select_budget_replacement(
                    cluster["members"],
                    support_threshold,
                    candidate_route,
                )
            )
            if (
                replacement_expert_id is None
                or not self._replace_budgeted_expert(
                    replacement_expert_id,
                    step,
                    replacement_audit,
                )
            ):
                return True, cluster_id, None, candidate_route

        # 选择代表
        if self.representative_mode == "historical_quality":
            representative = max(
                cluster["members"],
                key=lambda member: (
                    member["quality"]
                    * float(member.get("member_reliability", 1.0))
                ),
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
            "lifecycle_state": "active",
            "last_channel_support": None,
            "representative_mode": self.representative_mode,
            "admission_ttl_mode": self.admission_ttl_mode,
            "admission_route": candidate_route,
        }
        if self.candidate_reliability_enabled:
            expert["member_reliability"] = float(
                representative.get("member_reliability", 1.0)
            )
            expert["scoring_patch_reliabilities"] = [
                float(value)
                for value in torch.as_tensor(
                    representative.get(
                        "scoring_patch_reliabilities",
                        representative["patch_reliabilities"],
                    ),
                    dtype=torch.float32,
                ).tolist()
            ]
            expert["reference_reliability_penalty"] = float(
                self.reference_reliability_penalty
            )
            expert["reliability_audit"] = dict(
                representative.get("reliability_audit", {})
            )

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
        patch_reliabilities=None,
    ):
    # 连续性
        self.state_lifecycle_events = []
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
        if patch_reliabilities is not None:
            patch_reliabilities = torch.as_tensor(
                patch_reliabilities,
                dtype=torch.float32,
            ).detach().cpu()
            if patch_reliabilities.shape != (dino_patch_features.shape[0],):
                raise ValueError(
                    "patch_reliabilities must contain one value per patch"
                )
            if not torch.isfinite(patch_reliabilities).all():
                raise ValueError(
                    "patch_reliabilities contain non-finite values"
                )
            if (
                (patch_reliabilities <= 0.0)
                | (patch_reliabilities > 1.0)
            ).any():
                raise ValueError(
                    "patch_reliabilities must be within (0, 1]"
                )

        self._decay_temporary_states(step)

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
        self.last_budget_replacement_evaluation = None
        self.last_history_reentry_event = None
        self._record_coverage_observation(
            step,
            ms_score,
            ms_threshold,
            support_threshold,
            current_expert_supports,
            dino_patch_features=dino_patch_features,
            memory_patch_ids=memory_patch_ids,
            lab_feature=lab_feature,
        )

        (
            deleted_expert_ids,
            expert_lifecycle_events,
        ) = (
            self._advance_lifecycle(
                step,
                support_threshold,
                expert_channel_supports,
                current_expert_supports,
            )
        )

# 专家的候选和准入
        self.last_bootstrap_event = None
        if self.cold_start_mode == "confidence":
            bootstrap_expert_exists = any(
                int(expert["expert_id"])
                == int(self.bootstrap_confidence_expert_id)
                for expert in self.active_experts
                if self.bootstrap_confidence_expert_id is not None
            )
            if self.bootstrap_confidence_expert_id is None:
                (
                    is_candidate,
                    cluster_id,
                    admitted_expert_id,
                    candidate_route,
                ) = self._advance_confidence_cold_start(
                    step,
                    int(image_id),
                    dino_patch_features,
                    ms_score,
                    channel_support,
                    lab_feature,
                    patch_reliabilities,
                )
            elif not bootstrap_expert_exists:
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
                    patch_reliabilities=patch_reliabilities,
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
                    patch_reliabilities=patch_reliabilities,
                )
                has_non_bootstrap_expert = any(
                    expert.get("admission_route")
                    != "bootstrap_confidence"
                    and expert.get("lifecycle_state", "active") == "active"
                    for expert in self.active_experts
                )
                bootstrap_update_deferred = bool(
                    lab_drift_profile is not None
                    and (
                        lab_drift_profile.get("drift_confirmed", False)
                        or self.photometric_episode_active
                        or self.novel_quarantine_route
                        == "photometric_drift"
                    )
                )
                if not has_non_bootstrap_expert and not bootstrap_update_deferred:
                    self._advance_confidence_cold_start(
                        step,
                        int(image_id),
                        dino_patch_features,
                        ms_score,
                        channel_support,
                        lab_feature,
                        patch_reliabilities,
                    )
                elif not has_non_bootstrap_expert:
                    self.last_bootstrap_event = {
                        "decision": "confidence_state_update_deferred",
                        "reason": "photometric_state_hypothesis_active",
                        "step": int(step),
                        "image_id": int(image_id),
                        "drift_confirmed": bool(
                            lab_drift_profile.get("drift_confirmed", False)
                        ),
                        "photometric_episode_active": bool(
                            self.photometric_episode_active
                        ),
                        "photometric_pool_active": bool(
                            self.novel_quarantine_route
                            == "photometric_drift"
                        ),
                    }
        elif (
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
                provisional_channel_support,
                lab_feature,
                patch_reliabilities,
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
                patch_reliabilities=patch_reliabilities,
            )

        if self.last_budget_replacement is not None:
            replaced_expert_id = int(
                self.last_budget_replacement["expert_id"]
            )
            if replaced_expert_id not in deleted_expert_ids:
                deleted_expert_ids.append(replaced_expert_id)
            replacement_reason = self.last_budget_replacement.get(
                "reason", "coverage_budget_replacement"
            )
            replacement_decision = (
                "replaced_history_reentry"
                if replacement_reason == "history_reentry_swap"
                else "replaced_coreset_coverage"
                if replacement_reason in {
                    "coreset_coverage_replacement",
                    "coreset_minimum_coverage_loss_replacement",
                    "coreset_joint_minimum_loss_replacement",
                    "coreset_patch_victim_replacement",
                }
                else "replaced_coverage_budget"
            )
            expert_lifecycle_events.append(
                {
                    "step": int(step),
                    "expert_id": replaced_expert_id,
                    "decision": replacement_decision,
                    "signal_decision": replacement_decision,
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
            "budget_replacement_evaluation": (
                self.last_budget_replacement_evaluation
            ),
            "history_reentry": self.last_history_reentry_event,
            "fitness_history": self.last_fitness_history_profile,
            "historical_states": self._historical_state_audit(),
            "state_lifecycle_events": list(self.state_lifecycle_events),
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
                if expert.get("lifecycle_state", "active") == "active"
            ],
            "expert_slot_ids_after_step": [
                expert["expert_id"] for expert in self.active_experts
            ],
        }      

