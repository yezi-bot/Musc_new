import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.gradual_drift import (
    PHASES,
    GradualDriftDataset,
    build_balanced_stream_records,
    build_drift_schedule,
    build_lab_feature_stream,
    build_segment_balanced_stream_records,
)
from models.musc import MuSc


VARIANTS = {
    "cold_start_legacy": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "dual_path",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 8,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 4,
        "transition_fallback_min_hits": 2,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "legacy",
        "cold_start_lab_enabled": False,
    },
    "cold_start_bootstrap": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": False,
    },
    "cold_start_bootstrap_lab": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
    },
    "cold_start_bootstrap_lab_pool_audit": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "dual_path",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 8,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 4,
        "transition_fallback_min_hits": 2,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "drift_candidate_audit_enabled": True,
    },
    "cold_start_bootstrap_lab_photometric": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "photometric_candidate_enabled": True,
        "photometric_relevant_history_enabled": True,
    },
    "photometric_a_delayed_history": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "photometric_candidate_enabled": True,
        "photometric_relevant_history_enabled": True,
        "photometric_gate_mode": "coverage_loss",
        "photometric_candidate_window_mode": "history",
    },
    "photometric_b_early_fixed4": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "photometric_candidate_enabled": True,
        "photometric_relevant_history_enabled": False,
        "photometric_gate_mode": "lab_confirmed",
        "photometric_candidate_window_mode": "fixed_low_risk",
        "photometric_fixed_low_risk_members": 4,
    },
    "photometric_c_early_dynamic3_6": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "photometric_candidate_enabled": True,
        "photometric_relevant_history_enabled": False,
        "photometric_gate_mode": "lab_confirmed",
        "photometric_candidate_window_mode": "dynamic_low_risk",
        "photometric_dynamic_min_low_risk_members": 3,
        "photometric_dynamic_max_low_risk_members": 6,
    },
    "photometric_c_episode_exit": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "photometric_candidate_enabled": True,
        "photometric_relevant_history_enabled": False,
        "photometric_gate_mode": "lab_confirmed",
        "photometric_candidate_window_mode": "dynamic_low_risk",
        "photometric_dynamic_min_low_risk_members": 3,
        "photometric_dynamic_max_low_risk_members": 6,
        "photometric_episode_enabled": True,
        "photometric_exit_confirm_steps": 2,
        "photometric_candidate_max_age": 12,
        "minimum_active_experts": 0,
    },
    "photometric_c_episode_exit_min1": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "legacy",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 0,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 0,
        "transition_fallback_min_hits": 0,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
        "cold_start_mode": "adaptive",
        "cold_start_lab_enabled": True,
        "photometric_candidate_enabled": True,
        "photometric_relevant_history_enabled": False,
        "photometric_gate_mode": "lab_confirmed",
        "photometric_candidate_window_mode": "dynamic_low_risk",
        "photometric_dynamic_min_low_risk_members": 3,
        "photometric_dynamic_max_low_risk_members": 6,
        "photometric_episode_enabled": True,
        "photometric_exit_confirm_steps": 2,
        "photometric_candidate_max_age": 12,
        "minimum_active_experts": 1,
    },
    "dynamic_dino_channel_gate_rolling_fallback": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "dual_path",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 8,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 4,
        "transition_fallback_min_hits": 2,
        "transition_fallback_latch": True,
        "robust_shift_audit": False,
    },
    "dynamic_dino_channel_gate_robust_shift_audit": {
        "dynamic": True,
        "mode": "dino_only",
        "candidate_mode": "dual_path",
        "novel_admission_mode": "channel_gate",
        "transition_fallback_window": 8,
        "transition_fallback_require_pre_pool_expert": True,
        "transition_fallback_evidence_window": 4,
        "transition_fallback_min_hits": 2,
        "transition_fallback_latch": True,
        "robust_shift_audit": True,
        "robust_shift_max_patches": 128,
        "robust_shift_epsilon": 0.05,
        "robust_shift_iterations": 20,
        "robust_shift_window_size": 4,
        "robust_shift_window_keep": 3,
        "robust_shift_position_radius": 1,
    },
}

VARIANTS["photometric_c_episode_exit_cap2_reject"] = {
    **VARIANTS["photometric_c_episode_exit"],
    "committee_cap": 2,
    "photometric_budget_replacement": False,
    "legacy_budget_replacement": False,
}
VARIANTS["photometric_c_episode_exit_cap2_replace"] = {
    **VARIANTS["photometric_c_episode_exit"],
    "committee_cap": 2,
    "photometric_budget_replacement": True,
    "legacy_budget_replacement": False,
}
VARIANTS["photometric_multistage_cap2_reject"] = {
    **VARIANTS["photometric_c_episode_exit_cap2_reject"],
    "minimum_active_experts": 0,
    "lifecycle_mode": "dormant_replace",
}
VARIANTS["photometric_multistage_cap2_replace"] = {
    **VARIANTS["photometric_c_episode_exit_cap2_replace"],
    "minimum_active_experts": 0,
    "lifecycle_mode": "dormant_replace",
}
VARIANTS["photometric_multistage_cap2_dual_replace"] = {
    **VARIANTS["photometric_multistage_cap2_replace"],
    "photometric_budget_admission_mode": "channel_or_lab",
    "photometric_dynamic_min_low_risk_members": 3,
    "photometric_min_low_risk_members": 3,
}
VARIANTS["photometric_multistage_cap2_dual_replace_fast_reentry"] = {
    **VARIANTS["photometric_multistage_cap2_dual_replace"],
    "photometric_dynamic_min_low_risk_members": 2,
    "photometric_min_low_risk_members": 2,
}
VARIANTS["photometric_multistage_cap2_dual_replace_history_reentry"] = {
    **VARIANTS["photometric_multistage_cap2_dual_replace"],
    "photometric_history_reentry_enabled": True,
    "photometric_history_reentry_members": 2,
}
VARIANTS["photometric_multistage_cap2_adaptive_evidence"] = {
    **VARIANTS["photometric_multistage_cap2_dual_replace"],
    "photometric_history_reentry_enabled": True,
    "photometric_history_reentry_members": 2,
    "photometric_state_distribution_enabled": True,
    "photometric_temporary_state_enabled": True,
    "photometric_adaptive_candidate_evidence": True,
}
VARIANTS["photometric_multistage_cap2_adaptive_evidence_bootstrap_provisional"] = {
    **VARIANTS["photometric_multistage_cap2_adaptive_evidence"],
    "cold_start_provisional_enabled": True,
    "cold_start_provisional_confirm_steps": 2,
    "cold_start_provisional_max_wait_steps": 4,
    "cold_start_provisional_weight": 0.25,
    "cold_start_provisional_min_channel_support": 0.05,
}
VARIANTS["photometric_multistage_cap2_adaptive_evidence_bootstrap_confidence"] = {
    **VARIANTS["photometric_multistage_cap2_adaptive_evidence"],
    "cold_start_mode": "confidence",
    "cold_start_confidence_history": 16,
    "cold_start_provisional_enabled": False,
}
VARIANTS[
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential"
] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_confidence"
    ],
    "cold_start_reference_confidence_mode": "prequential",
}
VARIANTS[
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_stable_anchor"
] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential"
    ],
    "photometric_stable_lab_anchor_enabled": True,
}
VARIANTS[
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream"
] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential"
    ],
    "photometric_denstream_support_enabled": True,
    "photometric_denstream_decay": 0.9,
    "photometric_denstream_mature_support": 0.5,
    "photometric_denstream_discard_support": 0.05,
}
VARIANTS[
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness"
] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream"
    ],
    "photometric_fitness_history_enabled": True,
    "photometric_candidate_window_mode": "history",
}
VARIANTS[
    "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness_coreset"
] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness"
    ],
    "budget_coverage_loss_enabled": True,
}
VARIANTS["priority4_min_loss_coreset"] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness_coreset"
    ],
    "budget_coverage_min_loss_only": True,
}
VARIANTS["priority4_joint_coreset"] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness_coreset"
    ],
    "budget_coverage_joint_eviction_enabled": True,
}
VARIANTS["priority4_patch_coreset"] = {
    **VARIANTS["priority4_joint_coreset"],
    "budget_patch_coverage_enabled": True,
    "budget_patch_prototype_count": 32,
}
VARIANTS["priority4_patch_victim"] = {
    **VARIANTS[
        "photometric_multistage_cap2_adaptive_evidence_bootstrap_prequential_denstream_fitness"
    ],
    "budget_patch_coverage_enabled": True,
    "budget_patch_prototype_count": 32,
    "budget_patch_victim_selection_enabled": True,
}
VARIANTS["priority4_patch_kcenter_victim"] = {
    **VARIANTS["priority4_patch_victim"],
    "budget_patch_kcenter_victim_selection_enabled": True,
}
VARIANTS["priority5_task_aware_virtual_swap"] = {
    **VARIANTS["priority4_patch_kcenter_victim"],
    "budget_virtual_swap_enabled": True,
    "budget_virtual_swap_max_normal_score_shift": 0.05,
    "budget_virtual_swap_max_iqr_increase": 0.05,
    "budget_virtual_swap_history_size": 32,
}
VARIANTS["priority5_task_aware_soft_reliability"] = {
    **VARIANTS["priority5_task_aware_virtual_swap"],
    "candidate_reliability_enabled": True,
    "candidate_reliability_floor": 0.1,
    "candidate_unidirectional_weight": 0.25,
    "candidate_reliability_position_radius": 1,
    "reference_reliability_penalty": 1.0,
}
VARIANTS["priority5_task_aware_cross_patch_reliability"] = {
    **VARIANTS["priority5_task_aware_soft_reliability"],
    "candidate_patch_reliability_mode": "cross_patch_robust",
}
VARIANTS["photometric_multistage_cap2_state_distribution_only"] = {
    **VARIANTS["photometric_multistage_cap2_dual_replace_history_reentry"],
    "photometric_state_distribution_enabled": True,
    "photometric_temporary_state_enabled": False,
    "photometric_adaptive_candidate_evidence": False,
}
VARIANTS["photometric_multistage_cap2_state_distribution_temporary"] = {
    **VARIANTS["photometric_multistage_cap2_state_distribution_only"],
    "photometric_temporary_state_enabled": True,
}

PRIMARY_VARIANTS = [
    "cold_start_bootstrap",
    "cold_start_bootstrap_lab",
    "cold_start_bootstrap_lab_photometric",
]


DRIFT_LEVEL_SCALES = {
    "no_drift": 0.0,
    "light": 1.0,
    "medium": 2.0,
    "strong": 3.0,
}


LAB_FEATURE_SCALES = np.asarray(
    [100.0, 100.0, 128.0, 255.0, 128.0, 255.0],
    dtype=np.float64,
)


class GradualDriftMuSc(MuSc):
    def __init__(self, cfg, seed=0):
        self.stream_dataset = None
        super().__init__(cfg, seed=seed)

    def load_original_dataset(self, category):
        return super().load_datasets(
            category,
            divide_num=1,
            divide_iter=0,
        )

    def set_stream_dataset(self, dataset):
        self.stream_dataset = dataset

    def load_datasets(
        self,
        category,
        divide_num=1,
        divide_iter=0,
    ):
        if self.stream_dataset is not None:
            if divide_num != 1:
                raise ValueError(
                    "gradual drift stream requires "
                    "datasets.divide_num=1"
                )

            if divide_iter != 0:
                raise ValueError(
                    "gradual drift stream requires "
                    "divide_iter=0"
                )

            return self.stream_dataset

        return super().load_datasets(
            category,
            divide_num=divide_num,
            divide_iter=divide_iter,
        )


def load_config(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def safe_auroc(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    if labels.size == 0:
        return None

    if np.unique(labels).size < 2:
        return None

    return float(
        roc_auc_score(labels, scores)
    )


def select_stream_source_records(
    records,
    data_root,
    category,
    normal_source="test-good",
):
    test_records = [tuple(record) for record in records]
    test_normal_records = [
        record for record in test_records if record[1] == "good"
    ]
    anomaly_records = [
        record for record in test_records if record[1] != "good"
    ]
    train_normal_records = []
    if normal_source in {"train-good", "test-and-train-good"}:
        train_good_dir = (
            Path(data_root).resolve() / category / "train" / "good"
        )
        if not train_good_dir.is_dir():
            raise FileNotFoundError(
                f"MVTec train/good directory not found: {train_good_dir}"
            )
        train_normal_records = [
            (category, "good", str(image_path), None)
            for image_path in sorted(train_good_dir.iterdir())
            if image_path.is_file()
        ]
        if not train_normal_records:
            raise ValueError(
                f"no normal images found in {train_good_dir}"
            )

    if normal_source == "test-good":
        normal_records = test_normal_records
    elif normal_source == "train-good":
        normal_records = train_normal_records
    elif normal_source == "test-and-train-good":
        normal_records = test_normal_records + train_normal_records
    else:
        raise ValueError(f"unknown normal_source: {normal_source}")

    selected_records = normal_records + anomaly_records
    return selected_records, {
        "normal_source": normal_source,
        "available_test_good_count": len(test_normal_records),
        "available_train_good_count": len(train_normal_records),
        "available_selected_good_count": len(normal_records),
        "available_test_anomaly_count": len(anomaly_records),
    }


def configure_variant(
    base_config,
    args,
    variant_name,
    run_output,
):
    variant = VARIANTS[variant_name]
    cfg = copy.deepcopy(base_config)

    cfg["datasets"]["dataset_name"] = (
        "mvtec_ad"
    )
    cfg["datasets"]["data_path"] = str(
        Path(args.data_root).resolve()
    )
    cfg["datasets"]["class_name"] = (
        args.category
    )
    cfg["datasets"]["divide_num"] = 1

    cfg["testing"]["output_dir"] = str(
        run_output
    )
    cfg["testing"]["vis"] = False
    cfg["testing"]["save_excel"] = False
    cfg["testing"]["use_rscin"] = False
    cfg["testing"]["max_samples"] = None
    cfg["testing"]["shuffle_stream"] = False

    cfg["models"]["feature_layers"] = [23]
    cfg["models"]["r_list"] = [1]
    cfg["models"]["r_list2"] = [1]

    committee = cfg["models"].setdefault(
        "dynamic_committee",
        {},
    )

    committee["committee_cap"] = int(
        variant.get(
            "committee_cap",
            committee.get("committee_cap", 5),
        )
    )

    committee["enabled"] = variant[
        "dynamic"
    ]
    committee[
        "reset_at_category_boundaries"
    ] = False
    committee["candidate_mode"] = variant[
        "candidate_mode"
    ]
    committee[
        "representative_mode"
    ] = "historical_quality"
    committee[
        "admission_ttl_mode"
    ] = "historical_gap"
    committee[
        "novel_admission_mode"
    ] = variant["novel_admission_mode"]

    committee[
        "novel_quarantine_steps"
    ] = 8
    committee[
        "novel_quarantine_tail"
    ] = 4
    committee[
        "novel_ratio_threshold"
    ] = 0.95
    committee[
        "novel_provisional_weight"
    ] = 0.25
    committee[
        "novel_confirmed_weight"
    ] = 0.5
    committee[
        "novel_confirmed_ttl"
    ] = 16
    committee[
        "novel_memory_window"
    ] = 16
    committee[
        "novel_memory_keep_fraction"
    ] = 0.5
    committee[
        "memory_reliability_threshold"
    ] = 0.5
    committee[
        "novel_confirmed_members"
    ] = 1
    committee[
        "novel_ms_quantile"
    ] = 0.9
    committee[
        "novel_support_quantile"
    ] = 0.3
    committee[
        "novel_provisional_min"
    ] = 0.05
    committee[
        "ms_short_windows"
    ] = [8, 16, 32]
    committee["cold_start_mode"] = variant.get(
        "cold_start_mode",
        "legacy",
    )
    committee["cold_start_min_samples"] = 4
    committee["cold_start_max_samples"] = 6
    committee["cold_start_required_members"] = 3
    committee["cold_start_confidence_history"] = int(
        variant.get("cold_start_confidence_history", 16)
    )
    committee["cold_start_reference_confidence_mode"] = variant.get(
        "cold_start_reference_confidence_mode", "state"
    )
    committee["cold_start_lab_enabled"] = bool(
        variant.get("cold_start_lab_enabled", False)
    )
    committee["cold_start_provisional_enabled"] = bool(
        variant.get("cold_start_provisional_enabled", False)
    )
    committee["cold_start_provisional_confirm_steps"] = int(
        variant.get("cold_start_provisional_confirm_steps", 2)
    )
    committee["cold_start_provisional_max_wait_steps"] = int(
        variant.get("cold_start_provisional_max_wait_steps", 4)
    )
    committee["cold_start_provisional_weight"] = float(
        variant.get("cold_start_provisional_weight", 0.25)
    )
    committee["cold_start_provisional_min_channel_support"] = float(
        variant.get("cold_start_provisional_min_channel_support", 0.05)
    )
    committee["lab_ewma_alpha"] = args.lab_ewma_alpha
    committee["lab_ewma_threshold_k"] = args.lab_ewma_threshold_k
    committee["lab_ewma_confirm_steps"] = args.lab_ewma_confirm_steps
    committee["photometric_candidate_enabled"] = bool(
        variant.get("photometric_candidate_enabled", False)
    )
    committee["photometric_quarantine_steps"] = 4
    committee["photometric_prebuffer_size"] = 2
    committee["photometric_support_confirm_steps"] = (
        args.photometric_support_confirm_steps
    )
    committee["photometric_gate_mode"] = variant.get(
        "photometric_gate_mode",
        "coverage_loss",
    )
    committee["photometric_candidate_window_mode"] = variant.get(
        "photometric_candidate_window_mode",
        "history",
    )
    committee["photometric_fixed_low_risk_members"] = int(
        variant.get("photometric_fixed_low_risk_members", 4)
    )
    committee["photometric_dynamic_min_low_risk_members"] = int(
        variant.get("photometric_dynamic_min_low_risk_members", 3)
    )
    committee["photometric_dynamic_max_low_risk_members"] = int(
        variant.get("photometric_dynamic_max_low_risk_members", 6)
    )
    committee["photometric_episode_enabled"] = bool(
        variant.get("photometric_episode_enabled", False)
    )
    committee["photometric_exit_confirm_steps"] = int(
        variant.get("photometric_exit_confirm_steps", 2)
    )
    committee["photometric_candidate_max_age"] = int(
        variant.get("photometric_candidate_max_age", 12)
    )
    committee["photometric_local_lab_distance_threshold"] = 0.04
    committee["photometric_l_ewma_floor"] = args.photometric_l_ewma_floor
    committee["photometric_ab_ewma_floor"] = args.photometric_ab_ewma_floor
    committee["photometric_consistency_fraction"] = (
        args.photometric_consistency_fraction
    )
    committee["photometric_min_low_risk_members"] = int(
        variant.get("photometric_min_low_risk_members", 2)
    )
    committee["photometric_relevant_history_enabled"] = bool(
        variant.get("photometric_relevant_history_enabled", False)
    )
    committee["photometric_fitness_history_enabled"] = bool(
        variant.get("photometric_fitness_history_enabled", False)
    )
    committee["photometric_history_min"] = 3
    committee["photometric_history_max"] = 16
    state_registry_cap = variant.get("photometric_state_registry_cap")
    committee["photometric_state_registry_cap"] = (
        int(state_registry_cap)
        if state_registry_cap is not None
        else 2 * int(committee["committee_cap"])
    )
    committee["photometric_history_distance_threshold"] = 3.0
    committee["photometric_dino_similarity_floor"] = 0.9
    committee["photometric_history_reentry_enabled"] = bool(
        variant.get("photometric_history_reentry_enabled", False)
    )
    committee["photometric_history_reentry_members"] = int(
        variant.get("photometric_history_reentry_members", 2)
    )
    committee["photometric_history_median_risk_max"] = float(
        variant.get("photometric_history_median_risk_max", 1.25)
    )
    committee["photometric_history_extreme_risk_max"] = float(
        variant.get("photometric_history_extreme_risk_max", 2.0)
    )
    committee["photometric_adaptive_candidate_evidence"] = bool(
        variant.get("photometric_adaptive_candidate_evidence", False)
    )
    committee["photometric_stable_lab_anchor_enabled"] = bool(
        variant.get("photometric_stable_lab_anchor_enabled", False)
    )
    committee["photometric_denstream_support_enabled"] = bool(
        variant.get("photometric_denstream_support_enabled", False)
    )
    committee["photometric_denstream_decay"] = float(
        variant.get("photometric_denstream_decay", 0.9)
    )
    committee["photometric_denstream_mature_support"] = float(
        variant.get("photometric_denstream_mature_support", 0.5)
    )
    committee["photometric_denstream_discard_support"] = float(
        variant.get("photometric_denstream_discard_support", 0.05)
    )
    committee["photometric_state_distribution_enabled"] = bool(
        variant.get("photometric_state_distribution_enabled", False)
    )
    committee["photometric_temporary_state_enabled"] = bool(
        variant.get("photometric_temporary_state_enabled", False)
    )
    committee["photometric_budget_replacement"] = bool(
        variant.get("photometric_budget_replacement", True)
    )
    committee["photometric_budget_admission_mode"] = str(
        variant.get("photometric_budget_admission_mode", "channel_only")
    )
    committee["legacy_budget_replacement"] = bool(
        variant.get(
            "legacy_budget_replacement",
            committee.get("legacy_budget_replacement", True),
        )
    )
    committee["budget_coverage_loss_enabled"] = bool(
        variant.get("budget_coverage_loss_enabled", False)
    )
    committee["budget_coverage_min_loss_only"] = bool(
        variant.get("budget_coverage_min_loss_only", False)
    )
    committee["budget_coverage_joint_eviction_enabled"] = bool(
        variant.get("budget_coverage_joint_eviction_enabled", False)
    )
    committee["budget_patch_coverage_enabled"] = bool(
        variant.get("budget_patch_coverage_enabled", False)
    )
    committee["budget_patch_prototype_count"] = int(
        variant.get("budget_patch_prototype_count", 32)
    )
    committee["budget_patch_victim_selection_enabled"] = bool(
        variant.get("budget_patch_victim_selection_enabled", False)
    )
    committee["budget_patch_kcenter_victim_selection_enabled"] = bool(
        variant.get("budget_patch_kcenter_victim_selection_enabled", False)
    )
    committee["budget_virtual_swap_enabled"] = bool(
        variant.get("budget_virtual_swap_enabled", False)
    )
    committee["budget_virtual_swap_max_normal_score_shift"] = float(
        variant.get("budget_virtual_swap_max_normal_score_shift", 0.05)
    )
    committee["budget_virtual_swap_max_iqr_increase"] = float(
        variant.get("budget_virtual_swap_max_iqr_increase", 0.05)
    )
    committee["budget_virtual_swap_history_size"] = int(
        variant.get("budget_virtual_swap_history_size", 32)
    )
    committee["candidate_reliability_enabled"] = bool(
        variant.get("candidate_reliability_enabled", False)
    )
    committee["candidate_reliability_floor"] = float(
        variant.get("candidate_reliability_floor", 0.1)
    )
    committee["candidate_unidirectional_weight"] = float(
        variant.get("candidate_unidirectional_weight", 0.25)
    )
    committee["candidate_reliability_position_radius"] = int(
        variant.get("candidate_reliability_position_radius", 1)
    )
    committee["candidate_patch_reliability_mode"] = str(
        variant.get(
            "candidate_patch_reliability_mode",
            "legacy_rank_product",
        )
    )
    committee["reference_reliability_penalty"] = float(
        variant.get("reference_reliability_penalty", 1.0)
    )
    committee["minimum_active_experts"] = int(
        variant.get("minimum_active_experts", 0)
    )
    committee["lifecycle_mode"] = str(
        variant.get("lifecycle_mode", "ttl_delete")
    )
    committee["drift_candidate_audit_enabled"] = bool(
        variant.get("drift_candidate_audit_enabled", False)
    )
    committee["drift_candidate_pool_size"] = 4
    committee["drift_candidate_keep"] = 3
    committee["drift_lab_noise_floor"] = 0.01

    committee[
        "robust_shift_audit"
    ] = bool(
        variant.get(
            "robust_shift_audit",
            False,
        )
    )
    committee[
        "robust_shift_max_patches"
    ] = int(
        variant.get(
            "robust_shift_max_patches",
            128,
        )
    )
    committee[
        "robust_shift_epsilon"
    ] = float(
        variant.get(
            "robust_shift_epsilon",
            0.05,
        )
    )
    committee[
        "robust_shift_iterations"
    ] = int(
        variant.get(
            "robust_shift_iterations",
            20,
        )
    )
    committee[
        "robust_shift_window_size"
    ] = int(
        variant.get(
            "robust_shift_window_size",
            4,
        )
    )
    committee[
        "robust_shift_window_keep"
    ] = int(
        variant.get(
            "robust_shift_window_keep",
            3,
        )
    )
    committee[
        "robust_shift_position_radius"
    ] = variant.get(
        "robust_shift_position_radius",
        1,
    )

    fusion = cfg["models"].setdefault(
        "dynamic_fusion",
        {},
    )

    fusion["mode"] = variant["mode"]
    fusion[
        "transition_fallback_window"
    ] = int(
        variant[
            "transition_fallback_window"
        ]
    )
    fusion[
        "transition_fallback_require_pre_pool_expert"
    ] = bool(
        variant[
            "transition_fallback_require_pre_pool_expert"
        ]
    )
    fusion[
        "transition_fallback_evidence_window"
    ] = int(
        variant[
            "transition_fallback_evidence_window"
        ]
    )
    fusion[
        "transition_fallback_min_hits"
    ] = int(
        variant[
            "transition_fallback_min_hits"
        ]
    )
    fusion[
        "transition_fallback_latch"
    ] = bool(
        variant[
            "transition_fallback_latch"
        ]
    )

    return cfg


def load_audit(
    run_output,
    category,
    seed,
):
    audit_files = list(
        Path(run_output).rglob(
            f"{category}_dynamic_audit_"
            f"seed{seed}_part0.json"
        )
    )

    if len(audit_files) != 1:
        raise RuntimeError(
            f"expected one audit file, "
            f"found {len(audit_files)}"
        )

    with audit_files[0].open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def robust_values(record):
    robust_profile = (
        record.get(
            "robust_shift_profile"
        )
        or {}
    )

    nearest_expert_id = (
        robust_profile.get(
            "nearest_expert_id"
        )
    )
    robust_cost = (
        robust_profile.get(
            "nearest_discrete_cost"
        )
    )
    mutual_fraction = None

    if nearest_expert_id is not None:
        expert_profiles = (
            robust_profile.get(
                "expert_profiles",
                {},
            )
        )

        expert_profile = (
            expert_profiles.get(
                str(nearest_expert_id)
            )
        )

        if expert_profile is not None:
            mutual_fraction = (
                expert_profile.get(
                    "mutual_fraction"
                )
            )

    return {
        "nearest_expert_id": nearest_expert_id,
        "robust_cost": robust_cost,
        "mutual_fraction": mutual_fraction,
        "fixed_reference_expert_id": robust_profile.get(
            "fixed_reference_expert_id"
        ),
        "fixed_reference_image_id": robust_profile.get(
            "fixed_reference_image_id"
        ),
        "fixed_reference_cost": robust_profile.get(
            "fixed_reference_discrete_cost"
        ),
        "within_window_cost_median": robust_profile.get(
            "within_window_discrete_cost_median"
        ),
        "within_window_cost_max": robust_profile.get(
            "within_window_discrete_cost_max"
        ),
        "within_window_mutual_fraction_mean": robust_profile.get(
            "within_window_mutual_fraction_mean"
        ),
        "within_window_mutual_fraction_min": robust_profile.get(
            "within_window_mutual_fraction_min"
        ),
        "within_window_pair_count": robust_profile.get(
            "within_window_pair_count",
            0,
        ),
    }


def fixed_initial_expert(timeline):
    for record in timeline:
        admitted_expert_id = record.get(
            "admitted_expert_id"
        )
        if admitted_expert_id is None:
            continue
        admitted_expert = next(
            (
                expert
                for expert in record.get(
                    "active_experts_after",
                    [],
                )
                if int(expert["expert_id"])
                == int(admitted_expert_id)
            ),
            None,
        )
        if admitted_expert is None:
            continue
        return {
            "expert_id": int(admitted_expert_id),
            "image_id": int(admitted_expert["image_id"]),
            "admission_step": int(record["step"]),
        }
    return None


def _lab_distance_components(feature, reference_feature):
    delta = (
        np.asarray(feature, dtype=np.float64)
        - np.asarray(reference_feature, dtype=np.float64)
    ) / LAB_FEATURE_SCALES
    return {
        "l_distance": float(np.linalg.norm(delta[:2])),
        "ab_distance": float(np.linalg.norm(delta[2:])),
        "total_distance": float(np.linalg.norm(delta)),
    }


def build_lab_audit_profiles(
    lab_features,
    timeline=None,
    reference=None,
):
    lab_features = np.asarray(
        lab_features,
        dtype=np.float64,
    )
    if lab_features.ndim != 2 or lab_features.shape[1] != 6:
        raise ValueError(
            "lab_features must have shape [images, 6]"
        )
    if timeline is not None and lab_features.shape[0] != len(timeline):
        raise ValueError(
            "lab feature and timeline image counts must match"
        )

    if reference is None:
        if timeline is None:
            raise ValueError(
                "timeline or fixed reference is required"
            )
        reference = fixed_initial_expert(timeline)
    if reference is not None and not (
        0 <= int(reference["image_id"]) < lab_features.shape[0]
    ):
        raise ValueError(
            "fixed Lab reference image must be inside the stream"
        )
    if reference is not None and not (
        0 <= int(reference["admission_step"]) < lab_features.shape[0]
    ):
        raise ValueError(
            "fixed Lab reference admission step must be inside the stream"
        )
    if (
        reference is not None
        and int(reference["image_id"]) > int(reference["admission_step"])
    ):
        raise ValueError("fixed Lab reference must not come from the future")
    profiles = []
    for step, feature in enumerate(lab_features):
        profile = {
            "reference_expert_id": None,
            "reference_image_id": None,
            "reference_admission_step": None,
            "l_distance": None,
            "ab_distance": None,
            "total_distance": None,
        }
        if (
            reference is not None
            and step > reference["admission_step"]
        ):
            reference_feature = lab_features[
                reference["image_id"]
            ]
            distances = _lab_distance_components(
                feature,
                reference_feature,
            )
            profile.update(
                {
                    "reference_expert_id": reference["expert_id"],
                    "reference_image_id": reference["image_id"],
                    "reference_admission_step": reference[
                        "admission_step"
                    ],
                    **distances,
                }
            )
        profiles.append(profile)
    return profiles


def build_lab_ewma_profiles(
    lab_features,
    reference,
    alpha=0.3,
    threshold_k=3.0,
    baseline_window=8,
    confirm_steps=2,
):
    lab_features = np.asarray(lab_features, dtype=np.float64)
    if lab_features.ndim != 2 or lab_features.shape[1] != 6:
        raise ValueError("lab_features must have shape [images, 6]")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EWMA alpha must be within (0, 1]")
    if threshold_k < 0.0:
        raise ValueError("EWMA threshold_k must be non-negative")
    if baseline_window < 2:
        raise ValueError("EWMA baseline_window must be at least 2")
    if confirm_steps < 1:
        raise ValueError("EWMA confirm_steps must be positive")
    if reference is None:
        return [None for _ in range(lab_features.shape[0])]

    reference_image_id = int(reference["image_id"])
    admission_step = int(reference["admission_step"])
    if not 0 <= reference_image_id < lab_features.shape[0]:
        raise ValueError("EWMA reference image must be inside the stream")
    if not 0 <= admission_step < lab_features.shape[0]:
        raise ValueError("EWMA admission step must be inside the stream")
    if reference_image_id > admission_step:
        raise ValueError("EWMA reference must not come from the future")

    reference_feature = lab_features[reference_image_id]
    baseline_start = max(0, admission_step - baseline_window + 1)
    baseline_steps = list(range(baseline_start, admission_step + 1))
    baseline_profiles = [
        _lab_distance_components(
            lab_features[step],
            reference_feature,
        )
        for step in baseline_steps
    ]

    baseline = {}
    for name in ("l_distance", "ab_distance"):
        values = np.asarray(
            [profile[name] for profile in baseline_profiles],
            dtype=np.float64,
        )
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = max(1.4826 * mad, 1e-12)
        baseline[name] = {
            "median": median,
            "mad": mad,
            "threshold": float(median + threshold_k * robust_scale),
        }

    l_ewma = baseline["l_distance"]["median"]
    ab_ewma = baseline["ab_distance"]["median"]
    l_streak = 0
    ab_streak = 0
    drift_was_confirmed = False
    profiles = []
    for step, feature in enumerate(lab_features):
        if step <= admission_step:
            profiles.append(None)
            continue

        distances = _lab_distance_components(feature, reference_feature)
        l_ewma = float(
            alpha * distances["l_distance"]
            + (1.0 - alpha) * l_ewma
        )
        ab_ewma = float(
            alpha * distances["ab_distance"]
            + (1.0 - alpha) * ab_ewma
        )
        l_hit = l_ewma > baseline["l_distance"]["threshold"]
        ab_hit = ab_ewma > baseline["ab_distance"]["threshold"]
        l_streak = l_streak + 1 if l_hit else 0
        ab_streak = ab_streak + 1 if ab_hit else 0
        l_confirmed = l_streak >= confirm_steps
        ab_confirmed = ab_streak >= confirm_steps
        drift_confirmed = bool(l_confirmed or ab_confirmed)
        new_confirmation = drift_confirmed and not drift_was_confirmed
        profiles.append(
            {
                "baseline_start": baseline_start,
                "baseline_end": admission_step,
                "baseline_count": len(baseline_steps),
                "l_baseline_median": baseline["l_distance"]["median"],
                "l_baseline_mad": baseline["l_distance"]["mad"],
                "l_threshold": baseline["l_distance"]["threshold"],
                "ab_baseline_median": baseline["ab_distance"]["median"],
                "ab_baseline_mad": baseline["ab_distance"]["mad"],
                "ab_threshold": baseline["ab_distance"]["threshold"],
                "l_ewma": l_ewma,
                "ab_ewma": ab_ewma,
                "l_hit": bool(l_hit),
                "ab_hit": bool(ab_hit),
                "l_streak": int(l_streak),
                "ab_streak": int(ab_streak),
                "l_confirmed": bool(l_confirmed),
                "ab_confirmed": bool(ab_confirmed),
                "drift_confirmed": drift_confirmed,
                "new_confirmation": bool(new_confirmation),
            }
        )
        drift_was_confirmed = drift_confirmed
    return profiles


def lab_metrics(lab_profiles, schedule, labels):
    labels = np.asarray(labels)
    result = {}
    metric_names = (
        "l_distance",
        "ab_distance",
        "total_distance",
    )

    for phase in PHASES:
        for metric_name in metric_names:
            values = []
            clean_values = []
            for step, (profile, drift) in enumerate(
                zip(lab_profiles, schedule)
            ):
                if drift["phase"] != phase:
                    continue
                value = profile[metric_name]
                if value is None:
                    continue
                values.append(float(value))
                if not bool(labels[step]):
                    clean_values.append(float(value))
            result[
                f"{phase}_lab_{metric_name}_observations"
            ] = len(values)
            result[
                f"{phase}_lab_{metric_name}_median"
            ] = (
                float(np.median(values))
                if values
                else None
            )
            result[
                f"{phase}_lab_{metric_name}_clean_observations"
            ] = len(clean_values)
            result[
                f"{phase}_lab_{metric_name}_clean_median"
            ] = (
                float(np.median(clean_values))
                if clean_values
                else None
            )

    reference = next(
        (
            profile
            for profile in lab_profiles
            if profile["reference_expert_id"] is not None
        ),
        None,
    )
    result["lab_reference_expert_id"] = (
        reference["reference_expert_id"]
        if reference is not None
        else None
    )
    result["lab_reference_image_id"] = (
        reference["reference_image_id"]
        if reference is not None
        else None
    )
    result["lab_reference_admission_step"] = (
        reference["reference_admission_step"]
        if reference is not None
        else None
    )
    return result


def lab_ewma_metrics(ewma_profiles, schedule):
    result = {}
    first_confirmed_step = None
    first_confirmation_channel = None

    for phase in PHASES:
        phase_profiles = [
            profile
            for profile, drift in zip(ewma_profiles, schedule)
            if drift["phase"] == phase and profile is not None
        ]
        result[f"{phase}_lab_ewma_observations"] = len(phase_profiles)
        for channel in ("l", "ab"):
            values = [profile[f"{channel}_ewma"] for profile in phase_profiles]
            result[f"{phase}_lab_{channel}_ewma_median"] = (
                float(np.median(values)) if values else None
            )
            result[f"{phase}_lab_{channel}_ewma_max"] = (
                float(np.max(values)) if values else None
            )
            result[f"{phase}_lab_{channel}_hit_count"] = sum(
                bool(profile[f"{channel}_hit"])
                for profile in phase_profiles
            )
            result[f"{phase}_lab_{channel}_confirmed_count"] = sum(
                bool(profile[f"{channel}_confirmed"])
                for profile in phase_profiles
            )
        result[f"{phase}_lab_drift_confirmed_count"] = sum(
            bool(profile["drift_confirmed"])
            for profile in phase_profiles
        )

    for step, profile in enumerate(ewma_profiles):
        if profile is None or not profile["drift_confirmed"]:
            continue
        first_confirmed_step = step
        channels = []
        if profile["l_confirmed"]:
            channels.append("l")
        if profile["ab_confirmed"]:
            channels.append("ab")
        first_confirmation_channel = "+".join(channels)
        break

    result["lab_ewma_first_confirmed_step"] = first_confirmed_step
    result["lab_ewma_first_confirmation_channel"] = (
        first_confirmation_channel
    )
    result["lab_ewma_confirmed_step_count"] = sum(
        bool(profile and profile["drift_confirmed"])
        for profile in ewma_profiles
    )
    result["lab_ewma_confirmation_event_count"] = sum(
        bool(profile and profile["new_confirmation"])
        for profile in ewma_profiles
    )
    return result


def expert_lab_metrics(timeline, schedule):
    result = {}
    first_confirmed_step = None
    confirmed_step_count = 0
    observed_steps = 0
    stable_anchor_observed_steps = 0
    stable_anchor_frozen_steps = 0
    stable_anchor_change_count = 0
    candidate_state_observed_steps = 0
    candidate_anchor_distances = []
    coverage_anchor_distances = []
    previous_anchors = {}

    for phase in PHASES:
        phase_profiles = [
            record.get("lab_drift_profile")
            for record, drift in zip(timeline, schedule)
            if drift["phase"] == phase
            and record.get("lab_drift_profile") is not None
        ]
        confirmed_count = sum(
            bool(profile["drift_confirmed"])
            for profile in phase_profiles
        )
        result[f"{phase}_expert_lab_observations"] = len(phase_profiles)
        result[f"{phase}_expert_lab_confirmed_count"] = confirmed_count
        for channel in ("l", "ab"):
            values = [profile[f"{channel}_ewma"] for profile in phase_profiles]
            result[f"{phase}_expert_lab_{channel}_ewma_max"] = (
                float(np.max(values)) if values else None
            )
        observed_steps += len(phase_profiles)
        confirmed_step_count += confirmed_count

    for record in timeline:
        profile = record.get("lab_drift_profile")
        if profile is None:
            continue
        if profile["drift_confirmed"] and first_confirmed_step is None:
            first_confirmed_step = int(record["step"])
        if profile.get("stable_lab_anchor_enabled", False):
            stable_anchor_observed_steps += 1
            stable_anchor_frozen_steps += int(
                bool(profile.get("anchor_frozen", False))
            )
            expert_id = int(profile["expert_id"])
            anchor = np.asarray(
                profile["lab_anchor_center"], dtype=np.float64
            )
            previous = previous_anchors.get(expert_id)
            if previous is not None and not np.allclose(
                anchor,
                previous,
                rtol=0.0,
                atol=1e-12,
            ):
                stable_anchor_change_count += 1
            previous_anchors[expert_id] = anchor
            candidate_state_observed_steps += int(
                int(profile.get("candidate_lab_count", 0)) > 0
            )
            candidate_distance = profile.get(
                "anchor_to_candidate_total_distance"
            )
            if candidate_distance is not None:
                candidate_anchor_distances.append(float(candidate_distance))
            coverage_distance = profile.get(
                "anchor_to_coverage_total_distance"
            )
            if coverage_distance is not None:
                coverage_anchor_distances.append(float(coverage_distance))

    result["expert_lab_observed_steps"] = observed_steps
    result["expert_lab_first_confirmed_step"] = first_confirmed_step
    result["expert_lab_confirmed_step_count"] = confirmed_step_count
    result["stable_lab_anchor_observed_steps"] = (
        stable_anchor_observed_steps
    )
    result["stable_lab_anchor_frozen_steps"] = stable_anchor_frozen_steps
    result["stable_lab_anchor_change_count"] = stable_anchor_change_count
    result["lab_candidate_state_observed_steps"] = (
        candidate_state_observed_steps
    )
    result["lab_candidate_anchor_distance_max"] = (
        max(candidate_anchor_distances)
        if candidate_anchor_distances
        else None
    )
    result["lab_coverage_anchor_distance_max"] = (
        max(coverage_anchor_distances)
        if coverage_anchor_distances
        else None
    )
    return result


def photometric_route_metrics(timeline):
    confirmation_steps = [
        int(record["step"])
        for record in timeline
        if (record.get("lab_drift_profile") or {}).get(
            "just_confirmed", False
        )
    ]
    route_steps = [
        int(record["step"])
        for record in timeline
        if record.get("candidate_route") == "photometric_drift"
    ]
    quarantine_events = [
        record["novel_quarantine"]
        for record in timeline
        if (record.get("novel_quarantine") or {}).get("route")
        == "photometric_drift"
    ]
    admission_steps = [
        int(record["step"])
        for record in timeline
        if record.get("admission_route")
        == "photometric_drift_channel_gate"
    ]
    gate_trigger_steps = [
        int(record["step"])
        for record in timeline
        if (record.get("photometric_gate") or {}).get("triggered", False)
    ]
    episode_start_steps = [
        int(record["step"])
        for record in timeline
        if (record.get("photometric_gate") or {}).get(
            "episode_started", False
        )
    ]
    episode_end_steps = [
        int(record["step"])
        for record in timeline
        if (
            (record.get("photometric_gate") or {}).get(
                "episode_ended", False
            )
            or (record.get("novel_quarantine") or {}).get(
                "episode_finished", False
            )
        )
    ]
    abort_decisions = [
        str(event.get("decision", ""))
        for event in quarantine_events
        if str(event.get("decision", "")).startswith(
            "aborted_photometric"
        )
    ]
    coverage_floor_retained_count = sum(
        bool(event.get("coverage_floor_retained", False))
        for record in timeline
        for event in record.get("expert_lifecycle_events", [])
    )
    confirmed_patch_counts = [
        int(event["confirmed_patch_count"])
        for event in quarantine_events
        if event.get("confirmed_patch_count") is not None
    ]
    first_confirmation = confirmation_steps[0] if confirmation_steps else None
    first_admission = admission_steps[0] if admission_steps else None
    return {
        "photometric_lab_confirmation_count": len(confirmation_steps),
        "photometric_candidate_step_count": len(route_steps),
        "photometric_quarantine_open_count": sum(
            event.get("decision") == "opened"
            for event in quarantine_events
        ),
        "photometric_quarantine_confirm_count": sum(
            event.get("decision") == "confirmed_channel_gate"
            for event in quarantine_events
        ),
        "photometric_quarantine_reject_count": sum(
            str(event.get("decision", "")).startswith("rejected")
            for event in quarantine_events
        ),
        "photometric_quarantine_abort_count": len(abort_decisions),
        "photometric_recovery_abort_count": abort_decisions.count(
            "aborted_photometric_recovered"
        ),
        "photometric_reference_missing_abort_count": abort_decisions.count(
            "aborted_photometric_reference_missing"
        ),
        "photometric_timeout_abort_count": abort_decisions.count(
            "aborted_photometric_timeout"
        ),
        "photometric_admission_count": len(admission_steps),
        "photometric_gate_trigger_count": len(gate_trigger_steps),
        "photometric_episode_start_count": len(episode_start_steps),
        "photometric_episode_end_count": len(episode_end_steps),
        "coverage_floor_retained_count": coverage_floor_retained_count,
        "photometric_first_gate_trigger_step": (
            gate_trigger_steps[0] if gate_trigger_steps else None
        ),
        "photometric_first_confirmation_step": first_confirmation,
        "photometric_first_admission_step": first_admission,
        "photometric_confirmation_to_admission_delay": (
            first_admission - first_confirmation
            if first_confirmation is not None and first_admission is not None
            else None
        ),
        "photometric_confirmed_patch_count_median": (
            float(np.median(confirmed_patch_counts))
            if confirmed_patch_counts
            else None
        ),
    }


def drift_candidate_audit_metrics(
    timeline,
    schedule,
    labels,
    drift_level,
):
    labels = np.asarray(labels)
    completed = []
    missing_expert_count = 0
    for record in timeline:
        event = record.get("drift_candidate_audit") or {}
        decision = event.get("decision")
        if decision == "audited":
            selected_ids = [
                int(image_id)
                for image_id in event.get("selected_image_ids", [])
            ]
            selected_severities = [
                float(schedule[image_id]["severity"])
                for image_id in selected_ids
            ]
            severity_median = (
                float(np.median(selected_severities))
                if selected_severities
                else 0.0
            )
            completed.append(
                {
                    **event,
                    "severity_median": severity_median,
                    "is_true_drift": (
                        drift_level != "no_drift"
                        and severity_median > 0.0
                    ),
                    "selected_anomaly_count": sum(
                        bool(labels[image_id])
                        for image_id in selected_ids
                    ),
                }
            )
        elif decision == "audited_expert_missing":
            missing_expert_count += 1

    result = {
        "drift_pool_audit_event_count": len(completed),
        "drift_pool_audit_true_drift_count": sum(
            event["is_true_drift"] for event in completed
        ),
        "drift_pool_audit_zero_drift_count": sum(
            not event["is_true_drift"] for event in completed
        ),
        "drift_pool_audit_missing_expert_count": missing_expert_count,
        "drift_pool_audit_selected_anomaly_count": sum(
            event["selected_anomaly_count"] for event in completed
        ),
        "drift_pool_audit_selected_image_count": sum(
            len(event["selected_image_ids"]) for event in completed
        ),
    }
    metric_names = (
        "dino_consistency_min",
        "dino_to_trigger_expert_similarity",
        "lab_within_distance_median",
        "lab_within_distance_max",
        "lab_center_total_distance",
        "lab_separation_ratio",
        "channel_support_median",
        "mature_match_fraction_median",
        "recent_provisional_match_fraction_median",
    )
    for subset_name, subset in (
        ("all", completed),
        (
            "true_drift",
            [event for event in completed if event["is_true_drift"]],
        ),
        (
            "zero_drift",
            [event for event in completed if not event["is_true_drift"]],
        ),
    ):
        for metric_name in metric_names:
            values = [
                float(event[metric_name])
                for event in subset
                if event.get(metric_name) is not None
            ]
            result[f"drift_pool_{subset_name}_{metric_name}_median"] = (
                float(np.median(values)) if values else None
            )
    return result


def phase_metrics(
    labels,
    raw_scores,
    schedule,
):
    labels = np.asarray(labels)
    raw_scores = np.asarray(raw_scores)

    result = {}

    for phase in PHASES:
        indices = [
            index
            for index, drift in enumerate(
                schedule
            )
            if drift["phase"] == phase
        ]

        phase_labels = labels[indices]
        phase_scores = raw_scores[indices]

        normal_scores = phase_scores[
            phase_labels == 0
        ]

        result[
            f"{phase}_raw_auroc"
        ] = safe_auroc(
            phase_labels,
            phase_scores,
        )

        result[
            f"{phase}_normal_score_mean"
        ] = (
            float(
                np.mean(normal_scores)
            )
            if normal_scores.size > 0
            else None
        )

    return result


def state_member_contamination_metrics(timeline, labels):
    labels = np.asarray(labels)
    masses = []
    raw_fractions = []
    missing = 0
    invalid = 0
    for record in timeline:
        step = record.get("step")
        for expert in record.get("active_experts_before", []):
            ids = expert.get("state_member_image_ids")
            weights = expert.get("state_member_weights")
            if ids is None or weights is None:
                missing += 1
                continue
            if step is None:
                invalid += 1
                continue
            ids = np.asarray(ids)
            weights = np.asarray(weights, dtype=np.float64)
            if (
                ids.ndim != 1
                or weights.ndim != 1
                or len(ids) == 0
                or len(ids) != len(weights)
                or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0)
                or np.any(ids >= min(int(step), len(labels)))
                or not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or not np.isclose(weights.sum(), 1.0)
            ):
                invalid += 1
                continue
            anomaly_mask = labels[ids].astype(bool)
            masses.append(float(weights[anomaly_mask].sum()))
            raw_fractions.append(float(np.mean(anomaly_mask)))
    return {
        "state_member_weight_valid_snapshot_count": len(masses),
        "state_member_weight_missing_snapshot_count": missing,
        "state_member_weight_invalid_snapshot_count": invalid,
        "state_member_anomaly_mass_mean": (
            float(np.mean(masses)) if masses else None
        ),
        "state_member_anomaly_mass_max": (
            float(np.max(masses)) if masses else None
        ),
        "state_member_anomaly_raw_fraction_mean": (
            float(np.mean(raw_fractions)) if raw_fractions else None
        ),
        "state_member_anomaly_weight_reduction_mean": (
            float(np.mean(np.asarray(raw_fractions) - np.asarray(masses)))
            if masses
            else None
        ),
    }


def expert_metrics(
    timeline, labels, scoring=None, virtual_swap_enabled=False
):
    labels = np.asarray(labels)

    max_active_experts = 0
    final_active_experts = 0
    expert_admissions = 0
    expert_deletions = 0
    known_admission_sources = 0
    polluted_expert_admissions = 0
    expert_source_labels = {}
    expert_birth_steps = {}
    first_expert_admission_step = None
    first_expert_admission_route = None
    first_expert_image_id = None
    first_expert_is_anomaly = None
    bootstrap_selected_image_ids = []
    bootstrap_selected_anomaly_count = None
    bootstrap_provisional_created_count = 0
    bootstrap_provisional_confirming_step_count = 0
    bootstrap_provisional_rolled_back_count = 0
    bootstrap_provisional_admitted_count = 0
    bootstrap_provisional_confirmation_delays = []
    bootstrap_provisional_created_step = None
    bootstrap_confidence_created_count = 0
    bootstrap_confidence_update_count = 0
    bootstrap_confidence_first_positive_step = None
    bootstrap_confidence_values = []
    bootstrap_reference_confidence_first_positive_step = None
    bootstrap_reference_confidence_values = []
    bootstrap_normality_confidence_values = []
    bootstrap_prequential_support_values = []
    bootstrap_confidence_expert_ids = set()
    budget_replacement_count = 0
    legacy_budget_replacement_count = 0
    photometric_budget_replacement_count = 0
    channel_budget_admission_count = 0
    lab_budget_admission_count = 0
    channel_and_lab_budget_admission_count = 0
    redundant_lab_victim_count = 0
    replacement_new_coverage_fractions = []
    replacement_victim_unique_fractions = []
    budget_replacement_attempt_count = 0
    budget_replacement_evaluation_outcomes = {}
    evaluated_new_coverage_fractions = []
    evaluated_coverage_removal_losses = []
    evaluated_coverage_swap_gains = []
    evaluated_coverage_eviction_losses = []
    max_expert_slots = 0
    final_expert_slots = 0
    dormant_transition_count = 0
    expert_reactivation_count = 0
    history_reentry_admission_count = 0
    history_shared_coverage_count = 0
    history_restored_free_slot_count = 0
    history_restored_by_replacement_count = 0
    history_materialized_free_slot_count = 0
    history_materialized_by_replacement_count = 0
    history_reentry_rejected_count = 0
    max_historical_state_count = 0
    final_historical_state_count = 0
    adaptive_coverage_admission_count = 0
    photometric_state_only_count = 0
    state_lifecycle_counts = {
        event_type: 0
        for event_type in (
            "create",
            "match",
            "update",
            "merge",
            "mature",
            "evict",
        )
    }
    temporary_state_created_count = 0
    temporary_support_update_count = 0
    temporary_support_decay_count = 0
    temporary_decay_eviction_count = 0
    temporary_effective_support_values = []
    temporary_effective_support_final = None
    temporary_denstream_matured_count = 0
    fitness_history_available_counts = []
    fitness_history_selected_counts = []
    fitness_history_effective_sizes = []

    for record in timeline:
        active_after = record.get(
            "active_experts_after",
            [],
        )

        final_active_experts = len(
            active_after
        )
        max_active_experts = max(
            max_active_experts,
            final_active_experts,
        )
        expert_slots_after = record.get(
            "expert_slots_after", active_after
        )
        record_step = int(record.get("step", 0))
        for expert in expert_slots_after:
            expert_id = int(expert["expert_id"])
            image_id = int(expert["image_id"])
            expert_birth_steps.setdefault(expert_id, record_step)
            if 0 <= image_id < len(labels):
                expert_source_labels[expert_id] = bool(labels[image_id])
        final_expert_slots = len(expert_slots_after)
        max_expert_slots = max(max_expert_slots, final_expert_slots)
        for event in record.get("expert_lifecycle_events", []):
            dormant_transition_count += int(
                bool(event.get("became_dormant_this_step", False))
            )
            expert_reactivation_count += int(
                bool(event.get("reactivated_this_step", False))
            )
        quarantine_event = (
            record.get("novel_quarantine")
            or record.get("novel_quarantine_event")
            or {}
        )
        fitness_profile = record.get("fitness_history") or {}
        if fitness_profile:
            fitness_history_available_counts.append(
                int(fitness_profile.get("available_count", 0))
            )
            fitness_history_selected_counts.append(
                int(fitness_profile.get("selected_count", 0))
            )
            fitness_history_effective_sizes.append(
                float(fitness_profile.get("effective_history_size", 0.0))
            )
        photometric_state_only_count += int(
            quarantine_event.get("decision") in {
                "recorded_photometric_state_only",
                "matured_photometric_state_only",
            }
        )
        for state_event in record.get("state_lifecycle_events", []):
            event_type = str(state_event.get("event_type", ""))
            if event_type in state_lifecycle_counts:
                state_lifecycle_counts[event_type] += 1
            temporary_state_created_count += int(
                event_type == "create"
                and state_event.get("status_after") == "temporary"
            )
            if state_event.get("temporary_effective_support") is not None:
                temporary_effective_support_values.append(
                    float(state_event["temporary_effective_support"])
                )
            temporary_support_update_count += int(
                state_event.get("reason") == "denstream_effective_support"
            )
            temporary_support_decay_count += int(
                event_type == "decay"
                and state_event.get("reason")
                == "temporary_effective_support_decay"
            )
            temporary_decay_eviction_count += int(
                event_type == "evict"
                and state_event.get("reason")
                == "temporary_effective_support_decay"
            )
            temporary_denstream_matured_count += int(
                event_type == "mature"
                and str(state_event.get("reason", "")).startswith(
                    "denstream_"
                )
            )
        history_event = (
            record.get("history_reentry")
            or quarantine_event.get("history_reentry")
            or {}
        )
        history_decision = history_event.get("decision")
        history_shared_coverage_count += int(
            history_decision == "history_shared_coverage"
        )
        history_restored_free_slot_count += int(
            history_decision == "history_restored_free_slot"
        )
        history_restored_by_replacement_count += int(
            history_decision == "history_restored_by_replacement"
        )
        history_materialized_free_slot_count += int(
            history_decision == "history_materialized_free_slot"
        )
        history_materialized_by_replacement_count += int(
            history_decision == "history_materialized_by_replacement"
        )
        history_reentry_rejected_count += int(
            history_decision == "history_reentry_rejected"
        )
        max_historical_state_count = max(
            max_historical_state_count,
            len(record.get("historical_states", [])),
        )
        final_historical_state_count = len(
            record.get("historical_states", [])
        )
        final_temporary_supports = [
            float(state["temporary_effective_support"])
            for state in record.get("historical_states", [])
            if state.get("state_status") == "temporary"
            and state.get("temporary_effective_support") is not None
        ]
        temporary_effective_support_final = (
            max(final_temporary_supports)
            if final_temporary_supports
            else None
        )
        bootstrap_event = record.get("bootstrap") or {}
        bootstrap_decision = bootstrap_event.get("decision")
        if bootstrap_decision == "confidence_state_created":
            bootstrap_confidence_created_count += 1
        elif bootstrap_decision == "confidence_state_updated":
            bootstrap_confidence_update_count += 1
        if bootstrap_event.get("state_confidence") is not None:
            confidence = float(bootstrap_event["state_confidence"])
            bootstrap_confidence_values.append(confidence)
            if (
                confidence > 0.0
                and bootstrap_confidence_first_positive_step is None
            ):
                bootstrap_confidence_first_positive_step = record_step
        if bootstrap_event.get("reference_confidence") is not None:
            reference_confidence = float(
                bootstrap_event["reference_confidence"]
            )
            bootstrap_reference_confidence_values.append(
                reference_confidence
            )
            if (
                reference_confidence > 0.0
                and bootstrap_reference_confidence_first_positive_step is None
            ):
                bootstrap_reference_confidence_first_positive_step = (
                    record_step
                )
        if bootstrap_event.get("normality_confidence") is not None:
            bootstrap_normality_confidence_values.append(
                float(bootstrap_event["normality_confidence"])
            )
        if bootstrap_event.get("prequential_support") is not None:
            bootstrap_prequential_support_values.append(
                float(bootstrap_event["prequential_support"])
            )
        for expert in expert_slots_after:
            if expert.get("admission_route") == "bootstrap_confidence":
                bootstrap_confidence_expert_ids.add(int(expert["expert_id"]))
        if bootstrap_decision == "provisional_created":
            bootstrap_provisional_created_count += 1
            bootstrap_provisional_created_step = record_step
        elif bootstrap_decision == "provisional_confirming":
            bootstrap_provisional_confirming_step_count += 1
        elif bootstrap_decision == "rolled_back":
            bootstrap_provisional_rolled_back_count += 1
            bootstrap_provisional_created_step = None
        elif (
            bootstrap_decision == "admitted"
            and int(bootstrap_event.get("confirmation_steps", 0)) > 0
        ):
            bootstrap_provisional_admitted_count += 1
            if bootstrap_provisional_created_step is not None:
                bootstrap_provisional_confirmation_delays.append(
                    record_step - bootstrap_provisional_created_step
                )
            bootstrap_provisional_created_step = None

        deleted_ids = record.get(
            "deleted_expert_ids",
            [],
        )
        expert_deletions += len(
            deleted_ids
        )
        replacement = record.get("budget_replacement") or {}
        if replacement:
            budget_replacement_count += 1
            route = str(replacement.get("admission_route", ""))
            if route == "legacy":
                legacy_budget_replacement_count += 1
            elif route.startswith("photometric_drift"):
                photometric_budget_replacement_count += 1
            evidence = replacement.get("candidate_admission_evidence")
            if evidence == "channel":
                channel_budget_admission_count += 1
            elif evidence == "lab":
                lab_budget_admission_count += 1
            elif evidence == "channel_and_lab":
                channel_and_lab_budget_admission_count += 1
            if replacement.get("new_coverage_fraction") is not None:
                replacement_new_coverage_fractions.append(
                    float(replacement["new_coverage_fraction"])
                )
            victim_coverage = replacement.get("victim_coverage") or {}
            if victim_coverage.get("unique_fraction") is not None:
                replacement_victim_unique_fractions.append(
                    float(victim_coverage["unique_fraction"])
                )
            victim_lab = replacement.get("victim_lab_uniqueness") or {}
            redundant_lab_victim_count += int(
                bool(victim_lab.get("photometrically_redundant", False))
            )
        replacement_evaluation = (
            record.get("budget_replacement_evaluation") or {}
        )
        if replacement_evaluation:
            budget_replacement_attempt_count += 1
            outcome = str(
                replacement_evaluation.get("outcome", "unknown")
            )
            budget_replacement_evaluation_outcomes[outcome] = (
                budget_replacement_evaluation_outcomes.get(outcome, 0) + 1
            )
            evaluated_fraction = replacement_evaluation.get(
                "new_coverage_fraction"
            )
            if evaluated_fraction is not None:
                evaluated_new_coverage_fractions.append(
                    float(evaluated_fraction)
                )
            removal_loss = replacement_evaluation.get(
                "coverage_removal_loss"
            )
            if removal_loss is not None:
                evaluated_coverage_removal_losses.append(
                    float(removal_loss)
                )
            swap_gain = replacement_evaluation.get("coverage_swap_gain")
            if swap_gain is not None:
                evaluated_coverage_swap_gains.append(float(swap_gain))
            eviction_loss = replacement_evaluation.get(
                "coverage_eviction_loss"
            )
            if eviction_loss is not None:
                evaluated_coverage_eviction_losses.append(
                    float(eviction_loss)
                )

        admitted_expert_id = record.get(
            "admitted_expert_id"
        )

        if admitted_expert_id is None:
            continue

        expert_admissions += 1

        admitted_expert = next(
            (
                expert
                for expert in active_after
                if expert["expert_id"]
                == admitted_expert_id
            ),
            None,
        )

        if admitted_expert is None:
            continue

        if "history_reentry" in str(
            admitted_expert.get("admission_route", "")
        ):
            history_reentry_admission_count += 1
        if "adaptive_coverage" in str(
            admitted_expert.get("admission_route", "")
        ):
            adaptive_coverage_admission_count += 1

        image_id = int(
            admitted_expert["image_id"]
        )

        if first_expert_admission_step is None:
            first_expert_admission_step = int(record["step"])
            first_expert_admission_route = admitted_expert.get(
                "admission_route",
                record.get("admission_route"),
            )
            first_expert_image_id = image_id
            if 0 <= image_id < len(labels):
                first_expert_is_anomaly = bool(labels[image_id])
            bootstrap = record.get("bootstrap") or {}
            bootstrap_selected_image_ids = [
                int(selected_image_id)
                for selected_image_id in bootstrap.get(
                    "selected_image_ids",
                    [],
                )
            ]
            if bootstrap_selected_image_ids:
                bootstrap_selected_anomaly_count = sum(
                    bool(labels[selected_image_id])
                    for selected_image_id in bootstrap_selected_image_ids
                    if 0 <= selected_image_id < len(labels)
                )

        if not 0 <= image_id < len(labels):
            continue

        known_admission_sources += 1

        if bool(labels[image_id]):
            polluted_expert_admissions += 1

    polluted_fraction = (
        polluted_expert_admissions
        / known_admission_sources
        if known_admission_sources > 0
        else None
    )
    polluted_expert_scoring_step_count = None
    polluted_expert_future_anomaly_scoring_count = None
    bootstrap_provisional_scoring_step_count = None
    bootstrap_confidence_anomaly_exposure = None
    bootstrap_confidence_member_anomaly_exposure = None
    bootstrap_confidence_member_anomaly_scoring_step_count = None
    bootstrap_reference_first_weighted_scoring_step = None
    mean_reliability_score_shift = None
    effective_reference_weight_ratio = None
    polluted_effective_reference_weight = 0.0
    polluted_raw_reference_weight = 0.0
    polluted_actual_selection_step_count = 0
    polluted_reference_bank_fallback_step_count = 0
    polluted_effective_reference_weight_ratio = None
    polluted_reference_score_shift_mean = None
    polluted_reference_normal_score_shift_mean = None
    polluted_reference_anomaly_score_shift_mean = None
    if scoring is not None:
        polluted_scoring_steps = set()
        polluted_anomaly_steps = set()
        polluted_expert_ids = {
            expert_id
            for expert_id, is_anomaly in expert_source_labels.items()
            if is_anomaly
        }
        for scoring_index, scoring_record in enumerate(scoring):
            scoring_step = int(
                scoring_record.get("step", scoring_index)
            )
            participating_polluted_ids = {
                int(expert_id)
                for expert_id in scoring_record.get(
                    "active_expert_ids", []
                )
                if (
                    int(expert_id) in polluted_expert_ids
                    and scoring_step
                    > expert_birth_steps.get(int(expert_id), scoring_step)
                )
            }
            if not participating_polluted_ids:
                continue
            polluted_scoring_steps.add(scoring_step)
            if 0 <= scoring_step < len(labels) and bool(
                labels[scoring_step]
            ):
                polluted_anomaly_steps.add(scoring_step)
        polluted_expert_scoring_step_count = len(
            polluted_scoring_steps
        )
        polluted_expert_future_anomaly_scoring_count = len(
            polluted_anomaly_steps
        )
        bootstrap_provisional_scoring_step_count = sum(
            str(record.get("provisional_state", ""))
            == "bootstrap_provisional"
            for record in scoring
        )
        bootstrap_confidence_anomaly_exposure = 0.0
        bootstrap_confidence_member_anomaly_exposure = 0.0
        member_anomaly_scoring_steps = 0
        reliability_score_shifts = []
        effective_reference_weights = []
        raw_reference_weights = []
        polluted_effective_reference_weight = 0.0
        polluted_raw_reference_weight = 0.0
        polluted_actual_selection_step_count = 0
        polluted_reference_bank_fallback_step_count = 0
        polluted_reference_score_shifts = []
        polluted_reference_normal_score_shifts = []
        polluted_reference_anomaly_score_shifts = []
        for scoring_record in scoring:
            expert_ids = scoring_record.get("active_expert_ids", [])
            image_ids = scoring_record.get("expert_image_ids", [])
            weights = scoring_record.get(
                "active_expert_scoring_weights",
                [1.0] * len(expert_ids),
            )
            member_image_ids = scoring_record.get(
                "active_expert_member_image_ids",
                [[image_id] for image_id in image_ids],
            )
            member_weights = scoring_record.get(
                "active_expert_member_weights",
                [[] for _ in expert_ids],
            )
            score_shift = scoring_record.get(
                "reliability_score_shift_mean"
            )
            if score_shift is not None:
                reliability_score_shifts.append(float(score_shift))
                scoring_step = int(scoring_record.get("step", 0))
                has_polluted_reference = any(
                    int(expert_id) in polluted_expert_ids
                    and scoring_step
                    > expert_birth_steps.get(int(expert_id), scoring_step)
                    for expert_id in expert_ids
                )
                if has_polluted_reference:
                    polluted_reference_score_shifts.append(
                        float(score_shift)
                    )
                    if 0 <= scoring_step < len(labels):
                        target = (
                            polluted_reference_anomaly_score_shifts
                            if bool(labels[scoring_step])
                            else polluted_reference_normal_score_shifts
                        )
                        target.append(float(score_shift))
            effective_weight = scoring_record.get(
                "effective_reference_weight"
            )
            raw_weight = scoring_record.get("raw_reference_weight")
            if effective_weight is not None and raw_weight is not None:
                effective_reference_weights.append(float(effective_weight))
                raw_reference_weights.append(float(raw_weight))
            reference_contributions = scoring_record.get(
                "selected_reference_contributions", []
            )
            if reference_contributions:
                polluted_actual_selection_step_count += 1
                for contribution in reference_contributions:
                    if (
                        int(contribution["expert_id"])
                        in polluted_expert_ids
                    ):
                        polluted_effective_reference_weight += float(
                            contribution["effective_weight"]
                        )
                        polluted_raw_reference_weight += float(
                            contribution["raw_weight"]
                        )
            else:
                polluted_reference_bank_fallback_step_count += 1
                patch_reliability_means = scoring_record.get(
                    "active_expert_patch_reliability_means",
                    [1.0] * len(expert_ids),
                )
                for expert_id, patch_mean in zip(
                    expert_ids,
                    patch_reliability_means,
                ):
                    if int(expert_id) in polluted_expert_ids:
                        polluted_effective_reference_weight += float(
                            patch_mean
                        )
                        polluted_raw_reference_weight += 1.0
            if bootstrap_reference_first_weighted_scoring_step is None:
                if any(
                    int(expert_id) in bootstrap_confidence_expert_ids
                    and float(weight) > 0.0
                    for expert_id, weight in zip(expert_ids, weights)
                ):
                    bootstrap_reference_first_weighted_scoring_step = int(
                        scoring_record.get("step", 0)
                    )
            step_has_member_anomaly = False
            for expert_id, image_id, weight, ids, internal_weights in zip(
                expert_ids,
                image_ids,
                weights,
                member_image_ids,
                member_weights,
            ):
                if (
                    int(expert_id) in bootstrap_confidence_expert_ids
                    and 0 <= int(image_id) < len(labels)
                    and bool(labels[int(image_id)])
                ):
                    bootstrap_confidence_anomaly_exposure += float(weight)
                if int(expert_id) not in bootstrap_confidence_expert_ids:
                    continue
                if not internal_weights or len(internal_weights) != len(ids):
                    internal_weights = [1.0 / len(ids)] * len(ids) if ids else []
                anomaly_mass = sum(
                    float(member_weight)
                    for member_image_id, member_weight in zip(
                        ids, internal_weights
                    )
                    if (
                        0 <= int(member_image_id) < len(labels)
                        and bool(labels[int(member_image_id)])
                    )
                )
                if anomaly_mass > 0.0:
                    step_has_member_anomaly = True
                    bootstrap_confidence_member_anomaly_exposure += (
                        float(weight) * anomaly_mass
                    )
            member_anomaly_scoring_steps += int(step_has_member_anomaly)
        bootstrap_confidence_member_anomaly_scoring_step_count = (
            member_anomaly_scoring_steps
        )
        mean_reliability_score_shift = (
            float(np.mean(reliability_score_shifts))
            if reliability_score_shifts
            else None
        )
        effective_reference_weight_ratio = (
            float(
                np.sum(effective_reference_weights)
                / max(np.sum(raw_reference_weights), 1e-12)
            )
            if raw_reference_weights
            else None
        )
        polluted_effective_reference_weight_ratio = (
            float(
                polluted_effective_reference_weight
                / polluted_raw_reference_weight
            )
            if polluted_raw_reference_weight > 0.0
            else None
        )
        polluted_reference_score_shift_mean = (
            float(np.mean(polluted_reference_score_shifts))
            if polluted_reference_score_shifts
            else None
        )
        polluted_reference_normal_score_shift_mean = (
            float(np.mean(polluted_reference_normal_score_shifts))
            if polluted_reference_normal_score_shifts
            else None
        )
        polluted_reference_anomaly_score_shift_mean = (
            float(np.mean(polluted_reference_anomaly_score_shifts))
            if polluted_reference_anomaly_score_shifts
            else None
        )

    return {
        **state_member_contamination_metrics(timeline, labels),
        "max_active_experts": (
            max_active_experts
        ),
        "final_active_experts": (
            final_active_experts
        ),
        "max_expert_slots": max_expert_slots,
        "final_expert_slots": final_expert_slots,
        "dormant_transition_count": dormant_transition_count,
        "expert_reactivation_count": expert_reactivation_count,
        "history_reentry_admission_count": (
            history_reentry_admission_count
        ),
        "history_shared_coverage_count": history_shared_coverage_count,
        "history_restored_free_slot_count": (
            history_restored_free_slot_count
        ),
        "history_restored_by_replacement_count": (
            history_restored_by_replacement_count
        ),
        "history_materialized_free_slot_count": (
            history_materialized_free_slot_count
        ),
        "history_materialized_by_replacement_count": (
            history_materialized_by_replacement_count
        ),
        "history_reentry_rejected_count": history_reentry_rejected_count,
        "max_historical_state_count": max_historical_state_count,
        "final_historical_state_count": final_historical_state_count,
        "historical_state_created_count": state_lifecycle_counts["create"],
        "historical_state_matched_count": state_lifecycle_counts["match"],
        "historical_state_updated_count": state_lifecycle_counts["update"],
        "historical_state_merged_count": state_lifecycle_counts["merge"],
        "historical_state_matured_count": state_lifecycle_counts["mature"],
        "historical_state_evicted_count": state_lifecycle_counts["evict"],
        "temporary_state_created_count": temporary_state_created_count,
        "temporary_support_update_count": temporary_support_update_count,
        "temporary_support_decay_count": temporary_support_decay_count,
        "temporary_decay_eviction_count": temporary_decay_eviction_count,
        "temporary_denstream_matured_count": (
            temporary_denstream_matured_count
        ),
        "temporary_effective_support_max": (
            max(temporary_effective_support_values)
            if temporary_effective_support_values
            else None
        ),
        "temporary_effective_support_final": (
            temporary_effective_support_final
        ),
        "fitness_history_event_count": len(
            fitness_history_effective_sizes
        ),
        "mean_reliability_score_shift": mean_reliability_score_shift,
        "effective_reference_weight_ratio": (
            effective_reference_weight_ratio
        ),
        "polluted_effective_reference_weight": (
            polluted_effective_reference_weight
        ),
        "polluted_raw_reference_weight": polluted_raw_reference_weight,
        "polluted_actual_selection_step_count": (
            polluted_actual_selection_step_count
        ),
        "polluted_reference_bank_fallback_step_count": (
            polluted_reference_bank_fallback_step_count
        ),
        "polluted_effective_reference_weight_ratio": (
            polluted_effective_reference_weight_ratio
        ),
        "polluted_reference_score_shift_mean": (
            polluted_reference_score_shift_mean
        ),
        "polluted_reference_normal_score_shift_mean": (
            polluted_reference_normal_score_shift_mean
        ),
        "polluted_reference_anomaly_score_shift_mean": (
            polluted_reference_anomaly_score_shift_mean
        ),
        "fitness_history_available_mean": (
            float(np.mean(fitness_history_available_counts))
            if fitness_history_available_counts
            else None
        ),
        "fitness_history_selected_mean": (
            float(np.mean(fitness_history_selected_counts))
            if fitness_history_selected_counts
            else None
        ),
        "fitness_history_effective_size_mean": (
            float(np.mean(fitness_history_effective_sizes))
            if fitness_history_effective_sizes
            else None
        ),
        "adaptive_coverage_admission_count": (
            adaptive_coverage_admission_count
        ),
        "photometric_state_only_count": photometric_state_only_count,
        "expert_admissions": (
            expert_admissions
        ),
        "expert_deletions": (
            expert_deletions
        ),
        "budget_replacement_count": budget_replacement_count,
        "legacy_budget_replacement_count": (
            legacy_budget_replacement_count
        ),
        "photometric_budget_replacement_count": (
            photometric_budget_replacement_count
        ),
        "channel_budget_admission_count": channel_budget_admission_count,
        "lab_budget_admission_count": lab_budget_admission_count,
        "channel_and_lab_budget_admission_count": (
            channel_and_lab_budget_admission_count
        ),
        "redundant_lab_victim_count": redundant_lab_victim_count,
        "budget_replacement_new_coverage_fraction_median": (
            float(np.median(replacement_new_coverage_fractions))
            if replacement_new_coverage_fractions
            else None
        ),
        "budget_replacement_victim_unique_fraction_median": (
            float(np.median(replacement_victim_unique_fractions))
            if replacement_victim_unique_fractions
            else None
        ),
        "budget_replacement_attempt_count": (
            budget_replacement_attempt_count
        ),
        "budget_replacement_accepted_count": (
            budget_replacement_evaluation_outcomes.get("accepted", 0)
        ),
        "budget_replacement_disabled_count": (
            budget_replacement_evaluation_outcomes.get(
                "replacement_disabled", 0
            )
        ),
        "budget_replacement_insufficient_coverage_count": (
            budget_replacement_evaluation_outcomes.get(
                "insufficient_new_coverage", 0
            )
        ),
        "budget_replacement_missing_support_count": (
            budget_replacement_evaluation_outcomes.get(
                "missing_candidate_support", 0
            )
        ),
        "budget_replacement_no_redundant_victim_count": (
            budget_replacement_evaluation_outcomes.get(
                "no_redundant_victim", 0
            )
        ),
        "budget_replacement_no_positive_coverage_gain_count": (
            budget_replacement_evaluation_outcomes.get(
                "no_positive_coverage_gain", 0
            )
        ),
        "budget_replacement_candidate_evicted_count": (
            budget_replacement_evaluation_outcomes.get(
                "candidate_evicted_minimum_coverage_loss", 0
            )
        ),
        "budget_virtual_swap_enabled": bool(
            virtual_swap_enabled
        ),
        "budget_virtual_swap_rejected_count": (
            budget_replacement_evaluation_outcomes.get(
                "virtual_swap_rejected", 0
            )
        ),
        "budget_virtual_swap_insufficient_history_count": (
            budget_replacement_evaluation_outcomes.get(
                "virtual_swap_insufficient_history", 0
            )
        ),
        "budget_virtual_swap_no_positive_task_gain_count": (
            budget_replacement_evaluation_outcomes.get(
                "virtual_swap_no_positive_task_gain", 0
            )
        ),
        "budget_replacement_coverage_removal_loss_median": (
            float(np.median(evaluated_coverage_removal_losses))
            if evaluated_coverage_removal_losses
            else None
        ),
        "budget_replacement_coverage_swap_gain_median": (
            float(np.median(evaluated_coverage_swap_gains))
            if evaluated_coverage_swap_gains
            else None
        ),
        "budget_replacement_coverage_eviction_loss_median": (
            float(np.median(evaluated_coverage_eviction_losses))
            if evaluated_coverage_eviction_losses
            else None
        ),
        "budget_replacement_evaluated_new_coverage_fraction_median": (
            float(np.median(evaluated_new_coverage_fractions))
            if evaluated_new_coverage_fractions
            else None
        ),
        "known_admission_sources": (
            known_admission_sources
        ),
        "polluted_expert_admissions": (
            polluted_expert_admissions
        ),
        "polluted_admission_fraction": (
            polluted_fraction
        ),
        "polluted_expert_scoring_step_count": (
            polluted_expert_scoring_step_count
        ),
        "polluted_expert_future_anomaly_scoring_count": (
            polluted_expert_future_anomaly_scoring_count
        ),
        "first_expert_admission_step": first_expert_admission_step,
        "first_expert_admission_route": first_expert_admission_route,
        "first_expert_image_id": first_expert_image_id,
        "first_expert_is_anomaly": first_expert_is_anomaly,
        "bootstrap_selected_image_ids": json.dumps(
            bootstrap_selected_image_ids,
            ensure_ascii=False,
        ),
        "bootstrap_selected_anomaly_count": (
            bootstrap_selected_anomaly_count
        ),
        "bootstrap_provisional_created_count": (
            bootstrap_provisional_created_count
        ),
        "bootstrap_provisional_confirming_step_count": (
            bootstrap_provisional_confirming_step_count
        ),
        "bootstrap_provisional_rolled_back_count": (
            bootstrap_provisional_rolled_back_count
        ),
        "bootstrap_provisional_admitted_count": (
            bootstrap_provisional_admitted_count
        ),
        "bootstrap_provisional_scoring_step_count": (
            bootstrap_provisional_scoring_step_count
        ),
        "bootstrap_provisional_confirmation_delay_mean": (
            float(np.mean(bootstrap_provisional_confirmation_delays))
            if bootstrap_provisional_confirmation_delays
            else None
        ),
        "bootstrap_confidence_created_count": (
            bootstrap_confidence_created_count
        ),
        "bootstrap_confidence_update_count": (
            bootstrap_confidence_update_count
        ),
        "bootstrap_confidence_first_positive_step": (
            bootstrap_confidence_first_positive_step
        ),
        "bootstrap_confidence_final": (
            bootstrap_confidence_values[-1]
            if bootstrap_confidence_values
            else None
        ),
        "bootstrap_confidence_max": (
            max(bootstrap_confidence_values)
            if bootstrap_confidence_values
            else None
        ),
        "bootstrap_confidence_mean": (
            float(np.mean(bootstrap_confidence_values))
            if bootstrap_confidence_values
            else None
        ),
        "bootstrap_confidence_anomaly_exposure": (
            bootstrap_confidence_anomaly_exposure
        ),
        "bootstrap_confidence_member_anomaly_exposure": (
            bootstrap_confidence_member_anomaly_exposure
        ),
        "bootstrap_confidence_member_anomaly_scoring_step_count": (
            bootstrap_confidence_member_anomaly_scoring_step_count
        ),
        "bootstrap_reference_first_weighted_scoring_step": (
            bootstrap_reference_first_weighted_scoring_step
        ),
        "bootstrap_reference_confidence_first_positive_step": (
            bootstrap_reference_confidence_first_positive_step
        ),
        "bootstrap_reference_confidence_final": (
            bootstrap_reference_confidence_values[-1]
            if bootstrap_reference_confidence_values
            else None
        ),
        "bootstrap_reference_confidence_max": (
            max(bootstrap_reference_confidence_values)
            if bootstrap_reference_confidence_values
            else None
        ),
        "bootstrap_reference_confidence_mean": (
            float(np.mean(bootstrap_reference_confidence_values))
            if bootstrap_reference_confidence_values
            else None
        ),
        "bootstrap_normality_confidence_final": (
            bootstrap_normality_confidence_values[-1]
            if bootstrap_normality_confidence_values
            else None
        ),
        "bootstrap_normality_confidence_mean": (
            float(np.mean(bootstrap_normality_confidence_values))
            if bootstrap_normality_confidence_values
            else None
        ),
        "bootstrap_prequential_support_mean": (
            float(np.mean(bootstrap_prequential_support_values))
            if bootstrap_prequential_support_values
            else None
        ),
        "bootstrap_prequential_support_min": (
            min(bootstrap_prequential_support_values)
            if bootstrap_prequential_support_values
            else None
        ),
    }


def robust_metrics(
    timeline,
    schedule,
    labels,
):
    labels = np.asarray(labels)
    result = {}
    all_costs = []
    all_fixed_costs = []
    all_clean_fixed_costs = []
    all_within_medians = []
    all_within_maxes = []

    for phase in PHASES:
        phase_costs = []
        phase_fixed_costs = []
        phase_clean_fixed_costs = []
        phase_within_medians = []
        phase_within_maxes = []

        for record, drift in zip(
            timeline,
            schedule,
        ):
            if drift["phase"] != phase:
                continue

            values = robust_values(record)
            robust_cost = values["robust_cost"]
            fixed_cost = values["fixed_reference_cost"]
            within_median = values["within_window_cost_median"]
            within_max = values["within_window_cost_max"]

            if robust_cost is not None:
                phase_costs.append(
                    float(robust_cost)
                )
                all_costs.append(
                    float(robust_cost)
                )

            if fixed_cost is not None:
                phase_fixed_costs.append(float(fixed_cost))
                all_fixed_costs.append(float(fixed_cost))
                profile = record.get("robust_shift_profile") or {}
                selected_steps = profile.get("selected_steps", [])
                if selected_steps and all(
                    not bool(labels[int(selected_step)])
                    for selected_step in selected_steps
                ):
                    phase_clean_fixed_costs.append(float(fixed_cost))
                    all_clean_fixed_costs.append(float(fixed_cost))

            if within_median is not None:
                phase_within_medians.append(
                    float(within_median)
                )
                phase_within_maxes.append(
                    float(within_max)
                )
                all_within_medians.append(
                    float(within_median)
                )
                all_within_maxes.append(
                    float(within_max)
                )

        result[
            f"{phase}_robust_observations"
        ] = len(phase_costs)

        result[
            f"{phase}_robust_cost_median"
        ] = (
            float(
                np.median(phase_costs)
            )
            if phase_costs
            else None
        )
        result[
            f"{phase}_fixed_reference_observations"
        ] = len(phase_fixed_costs)
        result[
            f"{phase}_fixed_reference_cost_median"
        ] = (
            float(np.median(phase_fixed_costs))
            if phase_fixed_costs
            else None
        )
        result[
            f"{phase}_fixed_reference_clean_observations"
        ] = len(phase_clean_fixed_costs)
        result[
            f"{phase}_fixed_reference_clean_cost_median"
        ] = (
            float(np.median(phase_clean_fixed_costs))
            if phase_clean_fixed_costs
            else None
        )
        result[
            f"{phase}_within_window_observations"
        ] = len(phase_within_medians)
        result[
            f"{phase}_within_window_cost_median"
        ] = (
            float(np.median(phase_within_medians))
            if phase_within_medians
            else None
        )
        result[
            f"{phase}_within_window_cost_max_median"
        ] = (
            float(np.median(phase_within_maxes))
            if phase_within_maxes
            else None
        )

    result[
        "robust_shift_observed_steps"
    ] = len(all_costs)

    result[
        "robust_shift_cost_median"
    ] = (
        float(np.median(all_costs))
        if all_costs
        else None
    )
    result["fixed_reference_observed_steps"] = len(all_fixed_costs)
    result["fixed_reference_cost_median"] = (
        float(np.median(all_fixed_costs))
        if all_fixed_costs
        else None
    )
    result["fixed_reference_clean_observed_steps"] = len(
        all_clean_fixed_costs
    )
    result["fixed_reference_clean_cost_median"] = (
        float(np.median(all_clean_fixed_costs))
        if all_clean_fixed_costs
        else None
    )
    fixed_reference = next(
        (
            record.get("fixed_reference_expert_after")
            for record in timeline
            if record.get("fixed_reference_expert_after") is not None
        ),
        None,
    )
    result["fixed_reference_expert_id"] = (
        int(fixed_reference["expert_id"])
        if fixed_reference is not None
        else None
    )
    result["fixed_reference_image_id"] = (
        int(fixed_reference["image_id"])
        if fixed_reference is not None
        else None
    )
    result["fixed_reference_admission_step"] = (
        int(fixed_reference["admission_step"])
        if fixed_reference is not None
        else None
    )
    result[
        "within_window_observed_steps"
    ] = len(all_within_medians)
    result[
        "within_window_cost_median"
    ] = (
        float(np.median(all_within_medians))
        if all_within_medians
        else None
    )
    result[
        "within_window_cost_max_median"
    ] = (
        float(np.median(all_within_maxes))
        if all_within_maxes
        else None
    )

    return result


def write_step_diagnostics(
    output_path,
    stream_records,
    schedule,
    labels,
    raw_scores,
    timeline,
    lab_features,
    lab_profiles,
    lab_ewma_profiles,
):
    rows = []

    for step in range(len(timeline)):
        record = stream_records[step]
        drift = schedule[step]
        audit_record = timeline[step]
        lab_feature = lab_features[step]
        lab_profile = lab_profiles[step]
        lab_ewma_profile = lab_ewma_profiles[step] or {}
        expert_lab_profile = audit_record.get("lab_drift_profile") or {}

        robust = robust_values(audit_record)

        active_after = audit_record.get(
            "active_experts_after",
            [],
        )
        expert_slots_after = audit_record.get(
            "expert_slots_after", active_after
        )
        deleted_ids = audit_record.get(
            "deleted_expert_ids",
            [],
        )

        rows.append(
            {
                "step": step,
                "image_path": str(
                    record[2]
                ),
                "anomaly_type": record[1],
                "is_anomaly": int(
                    labels[step]
                ),
                "raw_image_score": float(
                    raw_scores[step]
                ),
                "drift_phase": drift[
                    "phase"
                ],
                "drift_segment_id": drift.get(
                    "segment_id",
                    drift["phase"],
                ),
                "drift_segment_kind": drift.get(
                    "segment_kind"
                ),
                "drift_state_id": drift.get(
                    "state_id"
                ),
                "drift_severity": drift[
                    "severity"
                ],
                "brightness": drift[
                    "brightness"
                ],
                "contrast": drift[
                    "contrast"
                ],
                "saturation": drift[
                    "saturation"
                ],
                "hue": drift["hue"],
                "lab_l_median": float(
                    lab_feature[0]
                ),
                "lab_l_iqr": float(
                    lab_feature[1]
                ),
                "lab_a_median": float(
                    lab_feature[2]
                ),
                "lab_a_iqr": float(
                    lab_feature[3]
                ),
                "lab_b_median": float(
                    lab_feature[4]
                ),
                "lab_b_iqr": float(
                    lab_feature[5]
                ),
                "lab_reference_expert_id": (
                    lab_profile[
                        "reference_expert_id"
                    ]
                ),
                "lab_reference_image_id": (
                    lab_profile[
                        "reference_image_id"
                    ]
                ),
                "lab_l_distance": (
                    lab_profile["l_distance"]
                ),
                "lab_ab_distance": (
                    lab_profile["ab_distance"]
                ),
                "lab_total_distance": (
                    lab_profile["total_distance"]
                ),
                "lab_ewma_baseline_start": (
                    lab_ewma_profile.get("baseline_start")
                ),
                "lab_ewma_baseline_end": (
                    lab_ewma_profile.get("baseline_end")
                ),
                "lab_ewma_baseline_count": (
                    lab_ewma_profile.get("baseline_count")
                ),
                "lab_l_baseline_median": (
                    lab_ewma_profile.get("l_baseline_median")
                ),
                "lab_l_baseline_mad": (
                    lab_ewma_profile.get("l_baseline_mad")
                ),
                "lab_l_threshold": (
                    lab_ewma_profile.get("l_threshold")
                ),
                "lab_ab_baseline_median": (
                    lab_ewma_profile.get("ab_baseline_median")
                ),
                "lab_ab_baseline_mad": (
                    lab_ewma_profile.get("ab_baseline_mad")
                ),
                "lab_ab_threshold": (
                    lab_ewma_profile.get("ab_threshold")
                ),
                "lab_l_ewma": lab_ewma_profile.get("l_ewma"),
                "lab_ab_ewma": lab_ewma_profile.get("ab_ewma"),
                "lab_l_hit": lab_ewma_profile.get("l_hit"),
                "lab_ab_hit": lab_ewma_profile.get("ab_hit"),
                "lab_l_streak": lab_ewma_profile.get("l_streak"),
                "lab_ab_streak": lab_ewma_profile.get("ab_streak"),
                "lab_l_confirmed": lab_ewma_profile.get("l_confirmed"),
                "lab_ab_confirmed": lab_ewma_profile.get("ab_confirmed"),
                "lab_drift_confirmed": lab_ewma_profile.get(
                    "drift_confirmed"
                ),
                "bootstrap_event": json.dumps(
                    audit_record.get("bootstrap"),
                    ensure_ascii=False,
                ),
                "drift_candidate_audit_event": json.dumps(
                    audit_record.get("drift_candidate_audit"),
                    ensure_ascii=False,
                ),
                "novel_quarantine_event": json.dumps(
                    audit_record.get("novel_quarantine"),
                    ensure_ascii=False,
                ),
                "fitness_history_event": json.dumps(
                    audit_record.get("fitness_history"),
                    ensure_ascii=False,
                ),
                "photometric_gate_event": json.dumps(
                    audit_record.get("photometric_gate"),
                    ensure_ascii=False,
                ),
                "budget_replacement_event": json.dumps(
                    audit_record.get("budget_replacement"),
                    ensure_ascii=False,
                ),
                "budget_replacement_evaluation": json.dumps(
                    audit_record.get("budget_replacement_evaluation"),
                    ensure_ascii=False,
                ),
                "expert_lab_expert_id": expert_lab_profile.get("expert_id"),
                "expert_lab_l_distance": expert_lab_profile.get(
                    "l_distance"
                ),
                "expert_lab_ab_distance": expert_lab_profile.get(
                    "ab_distance"
                ),
                "expert_lab_l_ewma": expert_lab_profile.get("l_ewma"),
                "expert_lab_ab_ewma": expert_lab_profile.get("ab_ewma"),
                "expert_lab_l_threshold": expert_lab_profile.get(
                    "l_threshold"
                ),
                "expert_lab_ab_threshold": expert_lab_profile.get(
                    "ab_threshold"
                ),
                "expert_lab_l_streak": expert_lab_profile.get("l_streak"),
                "expert_lab_ab_streak": expert_lab_profile.get("ab_streak"),
                "expert_lab_drift_confirmed": expert_lab_profile.get(
                    "drift_confirmed"
                ),
                "expert_lab_just_confirmed": expert_lab_profile.get(
                    "just_confirmed"
                ),
                "stable_lab_anchor_enabled": expert_lab_profile.get(
                    "stable_lab_anchor_enabled"
                ),
                "stable_lab_anchor_frozen": expert_lab_profile.get(
                    "anchor_frozen"
                ),
                "expert_lab_anchor_center": json.dumps(
                    expert_lab_profile.get("lab_anchor_center"),
                    ensure_ascii=False,
                ),
                "expert_lab_coverage_center": json.dumps(
                    expert_lab_profile.get("lab_coverage_center"),
                    ensure_ascii=False,
                ),
                "expert_lab_candidate_center": json.dumps(
                    expert_lab_profile.get("candidate_lab_center"),
                    ensure_ascii=False,
                ),
                "expert_lab_candidate_count": expert_lab_profile.get(
                    "candidate_lab_count"
                ),
                "expert_lab_anchor_to_coverage_distance": (
                    expert_lab_profile.get(
                        "anchor_to_coverage_total_distance"
                    )
                ),
                "expert_lab_anchor_to_candidate_distance": (
                    expert_lab_profile.get(
                        "anchor_to_candidate_total_distance"
                    )
                ),
                "ms_score": (
                    audit_record.get(
                        "ms_score"
                    )
                ),
                "channel_support": (
                    audit_record.get(
                        "channel_support"
                    )
                ),
                "candidate_route": (
                    audit_record.get(
                        "candidate_route"
                    )
                ),
                "admission_route": (
                    audit_record.get(
                        "admission_route"
                    )
                ),
                "admitted_expert_id": (
                    audit_record.get(
                        "admitted_expert_id"
                    )
                ),
                "support_threshold": audit_record.get(
                    "support_threshold"
                ),
                "current_expert_supports": json.dumps(
                    audit_record.get("current_expert_supports", {}),
                    ensure_ascii=False,
                ),
                "expert_lifecycle_events": json.dumps(
                    audit_record.get("expert_lifecycle_events", []),
                    ensure_ascii=False,
                ),
                "active_expert_ids_before": json.dumps(
                    [
                        expert["expert_id"]
                        for expert in audit_record.get(
                            "active_experts_before", []
                        )
                    ],
                    ensure_ascii=False,
                ),
                "expert_slot_ids_before": json.dumps(
                    [
                        expert["expert_id"]
                        for expert in audit_record.get(
                            "expert_slots_before", []
                        )
                    ],
                    ensure_ascii=False,
                ),
                "deleted_expert_ids": (
                    json.dumps(
                        deleted_ids,
                        ensure_ascii=False,
                    )
                ),
                "active_expert_count_after": (
                    len(active_after)
                ),
                "active_expert_ids_after": (
                    json.dumps(
                        [
                            expert[
                                "expert_id"
                            ]
                            for expert
                            in active_after
                        ],
                        ensure_ascii=False,
                    )
                ),
                "expert_slot_count_after": len(expert_slots_after),
                "dormant_expert_count_after": sum(
                    expert.get("lifecycle_state") == "dormant"
                    for expert in expert_slots_after
                ),
                "dormant_expert_ids_after": json.dumps(
                    [
                        expert["expert_id"]
                        for expert in expert_slots_after
                        if expert.get("lifecycle_state") == "dormant"
                    ],
                    ensure_ascii=False,
                ),
                "nearest_expert_id": (
                    robust["nearest_expert_id"]
                ),
                "fixed_reference_expert_id": (
                    robust["fixed_reference_expert_id"]
                ),
                "fixed_reference_image_id": (
                    robust["fixed_reference_image_id"]
                ),
                "fixed_reference_cost": (
                    robust["fixed_reference_cost"]
                ),
                "robust_shift_cost": (
                    robust["robust_cost"]
                ),
                "robust_mutual_fraction": (
                    robust["mutual_fraction"]
                ),
                "within_window_cost_median": (
                    robust["within_window_cost_median"]
                ),
                "within_window_cost_max": (
                    robust["within_window_cost_max"]
                ),
                "within_window_mutual_fraction_mean": (
                    robust["within_window_mutual_fraction_mean"]
                ),
                "within_window_mutual_fraction_min": (
                    robust["within_window_mutual_fraction_min"]
                ),
                "within_window_pair_count": (
                    robust["within_window_pair_count"]
                ),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def run_variant(
    base_config,
    args,
    drift_type,
    drift_level,
    seed,
    variant_name,
    schedule_override=None,
):
    drift_scale = (
        DRIFT_LEVEL_SCALES[drift_level]
        if schedule_override is None
        else 1.0
    )
    run_output = (
        Path(args.output_dir).resolve()
        / args.category
        / drift_type
        / drift_level
        / f"seed_{seed}"
        / variant_name
    )
    run_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = configure_variant(
        base_config=base_config,
        args=args,
        variant_name=variant_name,
        run_output=run_output,
    )

    model = GradualDriftMuSc(
        cfg,
        seed=seed,
    )

    base_dataset = (
        model.load_original_dataset(
            args.category
        )
    )

    if not hasattr(
        base_dataset,
        "data_to_iterate",
    ):
        raise RuntimeError(
            "expected MVTecDataset with "
            "data_to_iterate"
        )

    source_records, source_metadata = select_stream_source_records(
        records=base_dataset.data_to_iterate,
        data_root=args.data_root,
        category=args.category,
        normal_source=getattr(args, "normal_source", "test-good"),
    )

    if schedule_override is None:
        stream_records = build_balanced_stream_records(
            records=source_records,
            total_steps=args.stream_samples,
            anomaly_ratio=args.anomaly_ratio,
            seed=seed,
        )
        schedule = build_drift_schedule(
            drift_type=drift_type,
            total_steps=args.stream_samples,
            brightness_delta=args.brightness_delta * drift_scale,
            contrast_delta=args.contrast_delta * drift_scale,
            saturation_delta=args.saturation_delta * drift_scale,
            hue_delta=args.hue_delta * drift_scale,
        )
    else:
        schedule = [dict(item) for item in schedule_override]
        if len(schedule) != args.stream_samples:
            raise ValueError(
                "schedule_override length must match stream_samples"
            )
        segment_sizes = []
        previous_segment = None
        for item in schedule:
            segment_id = item.get("segment_id", item["phase"])
            if segment_id != previous_segment:
                segment_sizes.append(1)
                previous_segment = segment_id
            else:
                segment_sizes[-1] += 1
        stream_records = build_segment_balanced_stream_records(
            records=source_records,
            segment_sizes=segment_sizes,
            anomaly_ratio=args.anomaly_ratio,
            seed=seed,
            allow_reuse=bool(
                getattr(args, "allow_record_reuse", False)
            ),
        )

    drift_dataset = GradualDriftDataset(
        records=stream_records,
        transform_img=(
            base_dataset.transform_img
        ),
        transform_mask=(
            base_dataset.transform_mask
        ),
        drift_schedule=schedule,
    )

    model.set_stream_dataset(
        drift_dataset
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(
            model.device
        )

    started = time.perf_counter()

    (
        image_metric,
        pixel_metric,
        predictions,
    ) = model.make_category_data(
        args.category,
        return_predictions=True,
    )

    runtime_seconds = (
        time.perf_counter() - started
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(
            model.device
        )
        / 1024
        / 1024
        if torch.cuda.is_available()
        else 0.0
    )

    labels = np.asarray(
        predictions["image_labels"]
    )
    raw_scores = np.asarray(
        predictions["raw_image_scores"]
    )

    audit = load_audit(
        run_output=run_output,
        category=args.category,
        seed=seed,
    )
    timeline = audit["timeline"]

    if len(timeline) != len(
        stream_records
    ):
        raise RuntimeError(
            "timeline length does not "
            "match stream length"
        )

    lab_features = build_lab_feature_stream(
        stream_records,
        schedule,
    )
    lab_profiles = build_lab_audit_profiles(
        lab_features,
        timeline,
    )
    lab_reference = fixed_initial_expert(timeline)
    lab_ewma_profiles = build_lab_ewma_profiles(
        lab_features,
        reference=lab_reference,
        alpha=args.lab_ewma_alpha,
        threshold_k=args.lab_ewma_threshold_k,
        baseline_window=args.lab_ewma_baseline_window,
        confirm_steps=args.lab_ewma_confirm_steps,
    )

    write_step_diagnostics(
        output_path=(
            run_output
            / "gradual_drift_steps.csv"
        ),
        stream_records=stream_records,
        schedule=schedule,
        labels=labels,
        raw_scores=raw_scores,
        timeline=timeline,
        lab_features=lab_features,
        lab_profiles=lab_profiles,
        lab_ewma_profiles=lab_ewma_profiles,
    )

    row = {
        "category": args.category,
        **source_metadata,
        "drift_type": drift_type,
        "drift_level": drift_level,
        "drift_scale": drift_scale,
        "seed": seed,
        "variant": variant_name,
        "stream_samples": (
            args.stream_samples
        ),
        "anomaly_ratio": (
            args.anomaly_ratio
        ),
        "allow_record_reuse": bool(
            getattr(args, "allow_record_reuse", False)
        ),
        "unique_source_image_count": len(
            {str(record[2]) for record in stream_records}
        ),
        "reused_stream_record_count": (
            len(stream_records)
            - len({str(record[2]) for record in stream_records})
        ),
        "brightness_delta": (
            args.brightness_delta
            * drift_scale
        ),
        "contrast_delta": (
            args.contrast_delta
            * drift_scale
        ),
        "saturation_delta": (
            args.saturation_delta
            * drift_scale
        ),
        "hue_delta": (
            args.hue_delta
            * drift_scale
        ),
        "use_rscin": False,
        "image_auroc": float(
            image_metric[0]
        ),
        "pixel_auroc": float(
            pixel_metric[0]
        ),
        "aupro": float(
            pixel_metric[3]
        ),
        "raw_image_auroc": (
            safe_auroc(
                labels,
                raw_scores,
            )
        ),
        "runtime_seconds": (
            runtime_seconds
        ),
        "peak_memory_mb": (
            peak_memory_mb
        ),
        "robust_shift_audit": bool(
            cfg["models"][
                "dynamic_committee"
            ]["robust_shift_audit"]
        ),
        "candidate_mode": cfg["models"]["dynamic_committee"].get(
            "candidate_mode",
            "legacy",
        ),
        "transition_fallback_window": cfg["models"][
            "dynamic_committee"
        ].get("transition_fallback_window", 0),
        "transition_fallback_evidence_window": cfg["models"][
            "dynamic_committee"
        ].get("transition_fallback_evidence_window", 0),
        "transition_fallback_min_hits": cfg["models"][
            "dynamic_committee"
        ].get("transition_fallback_min_hits", 0),
        "lab_ewma_alpha": args.lab_ewma_alpha,
        "lab_ewma_threshold_k": args.lab_ewma_threshold_k,
        "lab_ewma_baseline_window": args.lab_ewma_baseline_window,
        "lab_ewma_confirm_steps": args.lab_ewma_confirm_steps,
        "cold_start_mode": cfg["models"]["dynamic_committee"].get(
            "cold_start_mode",
            "legacy",
        ),
        "cold_start_lab_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "cold_start_lab_enabled",
                False,
            )
        ),
        "cold_start_min_samples": cfg["models"]["dynamic_committee"].get(
            "cold_start_min_samples"
        ),
        "cold_start_max_samples": cfg["models"]["dynamic_committee"].get(
            "cold_start_max_samples"
        ),
        "cold_start_required_members": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_required_members"),
        "cold_start_confidence_history": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_confidence_history", 16),
        "cold_start_reference_confidence_mode": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_reference_confidence_mode", "state"),
        "cold_start_provisional_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "cold_start_provisional_enabled", False
            )
        ),
        "cold_start_provisional_confirm_steps": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_provisional_confirm_steps", 2),
        "cold_start_provisional_max_wait_steps": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_provisional_max_wait_steps", 4),
        "cold_start_provisional_weight": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_provisional_weight", 0.25),
        "cold_start_provisional_min_channel_support": cfg["models"][
            "dynamic_committee"
        ].get("cold_start_provisional_min_channel_support", 0.05),
        "photometric_candidate_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_candidate_enabled",
                False,
            )
        ),
        "photometric_quarantine_steps": cfg["models"][
            "dynamic_committee"
        ].get("photometric_quarantine_steps"),
        "photometric_prebuffer_size": cfg["models"][
            "dynamic_committee"
        ].get("photometric_prebuffer_size"),
        "photometric_support_confirm_steps": cfg["models"][
            "dynamic_committee"
        ].get("photometric_support_confirm_steps"),
        "photometric_gate_mode": cfg["models"]["dynamic_committee"].get(
            "photometric_gate_mode"
        ),
        "photometric_candidate_window_mode": cfg["models"][
            "dynamic_committee"
        ].get("photometric_candidate_window_mode"),
        "photometric_fixed_low_risk_members": cfg["models"][
            "dynamic_committee"
        ].get("photometric_fixed_low_risk_members"),
        "photometric_dynamic_min_low_risk_members": cfg["models"][
            "dynamic_committee"
        ].get("photometric_dynamic_min_low_risk_members"),
        "photometric_dynamic_max_low_risk_members": cfg["models"][
            "dynamic_committee"
        ].get("photometric_dynamic_max_low_risk_members"),
        "photometric_episode_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_episode_enabled", False
            )
        ),
        "photometric_exit_confirm_steps": cfg["models"][
            "dynamic_committee"
        ].get("photometric_exit_confirm_steps"),
        "photometric_candidate_max_age": cfg["models"][
            "dynamic_committee"
        ].get("photometric_candidate_max_age"),
        "minimum_active_experts": cfg["models"][
            "dynamic_committee"
        ].get("minimum_active_experts"),
        "lifecycle_mode": cfg["models"][
            "dynamic_committee"
        ].get("lifecycle_mode", "ttl_delete"),
        "committee_cap": cfg["models"][
            "dynamic_committee"
        ].get("committee_cap", 5),
        "legacy_budget_replacement": cfg["models"][
            "dynamic_committee"
        ].get("legacy_budget_replacement", True),
        "budget_coverage_window": cfg["models"][
            "dynamic_committee"
        ].get("budget_coverage_window", 32),
        "budget_min_new_coverage_fraction": cfg["models"][
            "dynamic_committee"
        ].get("budget_min_new_coverage_fraction", 0.5),
        "budget_replacement_margin": cfg["models"][
            "dynamic_committee"
        ].get("budget_replacement_margin", 0.0),
        "photometric_local_lab_distance_threshold": cfg["models"][
            "dynamic_committee"
        ].get("photometric_local_lab_distance_threshold"),
        "photometric_l_ewma_floor": cfg["models"][
            "dynamic_committee"
        ].get("photometric_l_ewma_floor"),
        "photometric_ab_ewma_floor": cfg["models"][
            "dynamic_committee"
        ].get("photometric_ab_ewma_floor"),
        "photometric_consistency_fraction": cfg["models"][
            "dynamic_committee"
        ].get("photometric_consistency_fraction"),
        "photometric_min_low_risk_members": cfg["models"][
            "dynamic_committee"
        ].get("photometric_min_low_risk_members"),
        "photometric_relevant_history_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_relevant_history_enabled", False
            )
        ),
        "photometric_fitness_history_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_fitness_history_enabled", False
            )
        ),
        "budget_coverage_loss_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_coverage_loss_enabled", False
            )
        ),
        "budget_coverage_min_loss_only": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_coverage_min_loss_only", False
            )
        ),
        "budget_coverage_joint_eviction_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_coverage_joint_eviction_enabled", False
            )
        ),
        "budget_patch_coverage_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_patch_coverage_enabled", False
            )
        ),
        "budget_patch_prototype_count": int(
            cfg["models"]["dynamic_committee"].get(
                "budget_patch_prototype_count", 32
            )
        ),
        "budget_patch_victim_selection_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_patch_victim_selection_enabled", False
            )
        ),
        "budget_patch_kcenter_victim_selection_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_patch_kcenter_victim_selection_enabled", False
            )
        ),
        "budget_virtual_swap_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "budget_virtual_swap_enabled", False
            )
        ),
        "budget_virtual_swap_max_normal_score_shift": float(
            cfg["models"]["dynamic_committee"].get(
                "budget_virtual_swap_max_normal_score_shift", 0.05
            )
        ),
        "budget_virtual_swap_max_iqr_increase": float(
            cfg["models"]["dynamic_committee"].get(
                "budget_virtual_swap_max_iqr_increase", 0.05
            )
        ),
        "budget_virtual_swap_history_size": int(
            cfg["models"]["dynamic_committee"].get(
                "budget_virtual_swap_history_size", 32
            )
        ),
        "candidate_reliability_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "candidate_reliability_enabled", False
            )
        ),
        "candidate_reliability_floor": cfg["models"][
            "dynamic_committee"
        ].get("candidate_reliability_floor"),
        "candidate_unidirectional_weight": cfg["models"][
            "dynamic_committee"
        ].get("candidate_unidirectional_weight"),
        "candidate_reliability_position_radius": cfg["models"][
            "dynamic_committee"
        ].get("candidate_reliability_position_radius"),
        "candidate_patch_reliability_mode": cfg["models"][
            "dynamic_committee"
        ].get("candidate_patch_reliability_mode"),
        "reference_reliability_penalty": cfg["models"][
            "dynamic_committee"
        ].get("reference_reliability_penalty"),
        "photometric_history_min": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_min"),
        "photometric_history_max": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_max"),
        "photometric_state_registry_cap": cfg["models"][
            "dynamic_committee"
        ].get("photometric_state_registry_cap"),
        "photometric_state_distribution_enabled": cfg["models"][
            "dynamic_committee"
        ].get("photometric_state_distribution_enabled", False),
        "photometric_temporary_state_enabled": cfg["models"][
            "dynamic_committee"
        ].get("photometric_temporary_state_enabled", False),
        "photometric_denstream_support_enabled": cfg["models"][
            "dynamic_committee"
        ].get("photometric_denstream_support_enabled", False),
        "photometric_denstream_decay": cfg["models"][
            "dynamic_committee"
        ].get("photometric_denstream_decay", 0.9),
        "photometric_denstream_mature_support": cfg["models"][
            "dynamic_committee"
        ].get("photometric_denstream_mature_support", 0.5),
        "photometric_denstream_discard_support": cfg["models"][
            "dynamic_committee"
        ].get("photometric_denstream_discard_support", 0.05),
        "photometric_history_distance_threshold": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_distance_threshold"),
        "photometric_dino_similarity_floor": cfg["models"][
            "dynamic_committee"
        ].get("photometric_dino_similarity_floor"),
        "photometric_history_reentry_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_history_reentry_enabled", False
            )
        ),
        "photometric_history_reentry_members": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_reentry_members"),
        "photometric_adaptive_candidate_evidence": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_adaptive_candidate_evidence", False
            )
        ),
        "photometric_stable_lab_anchor_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_stable_lab_anchor_enabled", False
            )
        ),
        "photometric_budget_replacement": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_budget_replacement", True
            )
        ),
        "photometric_budget_admission_mode": cfg["models"][
            "dynamic_committee"
        ].get("photometric_budget_admission_mode", "channel_only"),
        "drift_candidate_audit_enabled": bool(
            cfg["models"]["dynamic_committee"].get(
                "drift_candidate_audit_enabled",
                False,
            )
        ),
        "drift_candidate_pool_size": cfg["models"][
            "dynamic_committee"
        ].get("drift_candidate_pool_size"),
        "drift_candidate_keep": cfg["models"][
            "dynamic_committee"
        ].get("drift_candidate_keep"),
        "drift_lab_noise_floor": cfg["models"][
            "dynamic_committee"
        ].get("drift_lab_noise_floor"),
    }

    row.update(
        phase_metrics(
            labels=labels,
            raw_scores=raw_scores,
            schedule=schedule,
        )
    )

    row.update(
        expert_metrics(
            timeline=timeline,
            labels=labels,
            scoring=audit.get("scoring"),
            virtual_swap_enabled=bool(
                VARIANTS[variant_name].get(
                    "budget_virtual_swap_enabled", False
                )
            ),
        )
    )

    row.update(
        robust_metrics(
            timeline=timeline,
            schedule=schedule,
            labels=labels,
        )
    )

    row.update(
        lab_metrics(
            lab_profiles=lab_profiles,
            schedule=schedule,
            labels=labels,
        )
    )

    row.update(
        lab_ewma_metrics(
            ewma_profiles=lab_ewma_profiles,
            schedule=schedule,
        )
    )

    row.update(
        expert_lab_metrics(
            timeline=timeline,
            schedule=schedule,
        )
    )

    row.update(
        photometric_route_metrics(
            timeline=timeline,
        )
    )

    row.update(
        drift_candidate_audit_metrics(
            timeline=timeline,
            schedule=schedule,
            labels=labels,
            drift_level=drift_level,
        )
    )

    del drift_dataset
    del base_dataset
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row


def _is_monotonic(values):
    if len(values) < 2:
        return None, None
    non_decreasing = all(
        later >= earlier
        for earlier, later in zip(
            values,
            values[1:],
        )
    )
    strictly_increasing = all(
        later > earlier
        for earlier, later in zip(
            values,
            values[1:],
        )
    )
    return non_decreasing, strictly_increasing


def build_monotonicity_summary(
    rows,
    metric="plateau_robust_cost_median",
    require_robust_shift_audit=True,
):
    groups = {}
    for row in rows:
        if (
            require_robust_shift_audit
            and not row.get("robust_shift_audit")
        ):
            continue
        key = (
            row["category"],
            row["drift_type"],
            row["variant"],
        )
        groups.setdefault(key, []).append(row)

    reports = []
    for key, group_rows in sorted(groups.items()):
        category, drift_type, variant = key
        levels = []
        aggregate_values = []
        complete_aggregate = True
        for level, scale in DRIFT_LEVEL_SCALES.items():
            values = [
                float(row[metric])
                for row in group_rows
                if row.get("drift_level") == level
                and row.get(metric) is not None
            ]
            median_value = (
                float(np.median(values))
                if values
                else None
            )
            levels.append(
                {
                    "drift_level": level,
                    "drift_scale": scale,
                    "run_count": len(values),
                    "median_over_seeds": median_value,
                    "mean_over_seeds": (
                        float(np.mean(values))
                        if values
                        else None
                    ),
                    "std_over_seeds": (
                        float(np.std(values))
                        if values
                        else None
                    ),
                }
            )
            if median_value is None:
                complete_aggregate = False
            else:
                aggregate_values.append(median_value)

        if complete_aggregate:
            (
                aggregate_non_decreasing,
                aggregate_strictly_increasing,
            ) = _is_monotonic(aggregate_values)
        else:
            aggregate_non_decreasing = None
            aggregate_strictly_increasing = None

        per_seed = []
        seeds = sorted(
            {int(row["seed"]) for row in group_rows}
        )
        for seed in seeds:
            seed_values = []
            complete_seed = True
            for level in DRIFT_LEVEL_SCALES:
                matches = [
                    row.get(metric)
                    for row in group_rows
                    if int(row["seed"]) == seed
                    and row.get("drift_level") == level
                    and row.get(metric) is not None
                ]
                if len(matches) != 1:
                    complete_seed = False
                    break
                seed_values.append(float(matches[0]))
            if complete_seed:
                (
                    seed_non_decreasing,
                    seed_strictly_increasing,
                ) = _is_monotonic(seed_values)
            else:
                seed_non_decreasing = None
                seed_strictly_increasing = None
            per_seed.append(
                {
                    "seed": seed,
                    "values": (
                        seed_values
                        if complete_seed
                        else None
                    ),
                    "non_decreasing": seed_non_decreasing,
                    "strictly_increasing": seed_strictly_increasing,
                }
            )

        complete_seed_results = [
            item["non_decreasing"]
            for item in per_seed
            if item["non_decreasing"] is not None
        ]
        reports.append(
            {
                "category": category,
                "drift_type": drift_type,
                "variant": variant,
                "metric": metric,
                "levels": levels,
                "aggregate_non_decreasing": aggregate_non_decreasing,
                "aggregate_strictly_increasing": (
                    aggregate_strictly_increasing
                ),
                "all_complete_seeds_non_decreasing": (
                    all(complete_seed_results)
                    if complete_seed_results
                    else None
                ),
                "complete_seed_count": len(complete_seed_results),
                "per_seed": per_seed,
            }
        )
    return reports


def write_results(
    output_dir,
    rows,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "gradual_drift_runs.csv"
    )
    json_path = (
        output_dir
        / "gradual_drift_summary.json"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    nearest_monotonicity = build_monotonicity_summary(rows)
    fixed_monotonicity = build_monotonicity_summary(
        rows,
        metric="plateau_fixed_reference_cost_median",
    )
    fixed_clean_monotonicity = build_monotonicity_summary(
        rows,
        metric="plateau_fixed_reference_clean_cost_median",
    )
    lab_l_monotonicity = build_monotonicity_summary(
        rows,
        metric="plateau_lab_l_distance_median",
        require_robust_shift_audit=False,
    )
    lab_ab_monotonicity = build_monotonicity_summary(
        rows,
        metric="plateau_lab_ab_distance_median",
        require_robust_shift_audit=False,
    )
    lab_total_monotonicity = build_monotonicity_summary(
        rows,
        metric="plateau_lab_total_distance_median",
        require_robust_shift_audit=False,
    )
    lab_total_clean_monotonicity = build_monotonicity_summary(
        rows,
        metric="plateau_lab_total_distance_clean_median",
        require_robust_shift_audit=False,
    )
    json_path.write_text(
        json.dumps(
            {
                "runs": rows,
                "nearest_expert_distance_monotonicity": (
                    nearest_monotonicity
                ),
                "fixed_initial_expert_distance_monotonicity": (
                    fixed_monotonicity
                ),
                "fixed_initial_expert_clean_distance_monotonicity": (
                    fixed_clean_monotonicity
                ),
                "lab_l_distance_monotonicity": (
                    lab_l_monotonicity
                ),
                "lab_ab_distance_monotonicity": (
                    lab_ab_monotonicity
                ),
                "lab_total_distance_monotonicity": (
                    lab_total_monotonicity
                ),
                "lab_total_clean_distance_monotonicity": (
                    lab_total_clean_monotonicity
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate dynamic experts "
            "under gradual brightness "
            "and color drift."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            PROJECT_ROOT
            / "configs"
            / "musc.yaml"
        ),
    )
    parser.add_argument(
        "--data-root",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--category",
        default="bottle",
    )
    parser.add_argument(
        "--drift-types",
        nargs="+",
        choices=[
            "brightness",
            "color",
        ],
        default=[
            "brightness",
            "color",
        ],
    )
    parser.add_argument(
        "--drift-levels",
        nargs="+",
        choices=list(DRIFT_LEVEL_SCALES),
        default=list(DRIFT_LEVEL_SCALES),
        help=(
            "漂移强度档位；各档分别使用基础 delta 的 "
            "0、1、2、3 倍。"
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "--stream-samples",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=(1.0 / 3.0),
    )
    parser.add_argument(
        "--brightness-delta",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--contrast-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--saturation-delta",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--hue-delta",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--lab-ewma-alpha",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--lab-ewma-threshold-k",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--lab-ewma-baseline-window",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lab-ewma-confirm-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--photometric-support-confirm-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--photometric-l-ewma-floor",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--photometric-ab-ewma-floor",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--photometric-consistency-fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANTS),
        default=PRIMARY_VARIANTS,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not 0.0 < args.lab_ewma_alpha <= 1.0:
        raise ValueError("lab-ewma-alpha must be within (0, 1]")
    if args.lab_ewma_threshold_k < 0.0:
        raise ValueError("lab-ewma-threshold-k must be non-negative")
    if args.lab_ewma_baseline_window < 2:
        raise ValueError("lab-ewma-baseline-window must be at least 2")
    if args.lab_ewma_confirm_steps < 1:
        raise ValueError("lab-ewma-confirm-steps must be positive")
    if args.photometric_support_confirm_steps < 1:
        raise ValueError(
            "photometric-support-confirm-steps must be positive"
        )
    if args.photometric_l_ewma_floor <= 0.0:
        raise ValueError("photometric-l-ewma-floor must be positive")
    if args.photometric_ab_ewma_floor <= 0.0:
        raise ValueError("photometric-ab-ewma-floor must be positive")
    if not 0.0 < args.photometric_consistency_fraction <= 1.0:
        raise ValueError(
            "photometric-consistency-fraction must be within (0, 1]"
        )

    if args.stream_samples < 10:
        raise ValueError(
            "stream-samples must be "
            "at least 10"
        )

    if (
        args.stream_samples
        % len(PHASES)
        != 0
    ):
        raise ValueError(
            "stream-samples must be "
            "divisible by 5"
        )

    base_config = load_config(
        args.config
    )
    rows = []

    for drift_type in args.drift_types:
        for drift_level in args.drift_levels:
            for seed in args.seeds:
                for variant_name in args.variants:
                    print(
                        f"Running "
                        f"category={args.category}, "
                        f"drift={drift_type}, "
                        f"level={drift_level}, "
                        f"seed={seed}, "
                        f"variant={variant_name}"
                    )

                    row = run_variant(
                        base_config=base_config,
                        args=args,
                        drift_type=drift_type,
                        drift_level=drift_level,
                        seed=seed,
                        variant_name=variant_name,
                    )
                    rows.append(row)

                    write_results(
                        output_dir=(
                            args.output_dir
                        ),
                        rows=rows,
                    )


if __name__ == "__main__":
    main()
