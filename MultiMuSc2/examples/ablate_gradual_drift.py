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
    committee["cold_start_lab_enabled"] = bool(
        variant.get("cold_start_lab_enabled", False)
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
    committee["photometric_min_low_risk_members"] = 2
    committee["photometric_relevant_history_enabled"] = bool(
        variant.get("photometric_relevant_history_enabled", False)
    )
    committee["photometric_history_min"] = 3
    committee["photometric_history_max"] = 16
    committee["photometric_history_distance_threshold"] = 3.0
    committee["photometric_dino_similarity_floor"] = 0.9
    committee["photometric_budget_replacement"] = bool(
        variant.get("photometric_budget_replacement", True)
    )
    committee["legacy_budget_replacement"] = bool(
        variant.get(
            "legacy_budget_replacement",
            committee.get("legacy_budget_replacement", True),
        )
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
        if profile is not None and profile["drift_confirmed"]:
            first_confirmed_step = int(record["step"])
            break

    result["expert_lab_observed_steps"] = observed_steps
    result["expert_lab_first_confirmed_step"] = first_confirmed_step
    result["expert_lab_confirmed_step_count"] = confirmed_step_count
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
        if (record.get("photometric_gate") or {}).get(
            "episode_ended", False
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


def expert_metrics(timeline, labels):
    labels = np.asarray(labels)

    max_active_experts = 0
    final_active_experts = 0
    expert_admissions = 0
    expert_deletions = 0
    known_admission_sources = 0
    polluted_expert_admissions = 0
    first_expert_admission_step = None
    first_expert_admission_route = None
    first_expert_image_id = None
    first_expert_is_anomaly = None
    bootstrap_selected_image_ids = []
    bootstrap_selected_anomaly_count = None
    budget_replacement_count = 0
    legacy_budget_replacement_count = 0
    photometric_budget_replacement_count = 0
    replacement_new_coverage_fractions = []
    replacement_victim_unique_fractions = []
    budget_replacement_attempt_count = 0
    budget_replacement_evaluation_outcomes = {}
    evaluated_new_coverage_fractions = []
    max_expert_slots = 0
    final_expert_slots = 0
    dormant_transition_count = 0
    expert_reactivation_count = 0

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
        final_expert_slots = len(expert_slots_after)
        max_expert_slots = max(max_expert_slots, final_expert_slots)
        for event in record.get("expert_lifecycle_events", []):
            dormant_transition_count += int(
                bool(event.get("became_dormant_this_step", False))
            )
            expert_reactivation_count += int(
                bool(event.get("reactivated_this_step", False))
            )

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
            if replacement.get("new_coverage_fraction") is not None:
                replacement_new_coverage_fractions.append(
                    float(replacement["new_coverage_fraction"])
                )
            victim_coverage = replacement.get("victim_coverage") or {}
            if victim_coverage.get("unique_fraction") is not None:
                replacement_victim_unique_fractions.append(
                    float(victim_coverage["unique_fraction"])
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

    return {
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
        Path(args.output_dir)
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

    if schedule_override is None:
        stream_records = build_balanced_stream_records(
            records=base_dataset.data_to_iterate,
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
            records=base_dataset.data_to_iterate,
            segment_sizes=segment_sizes,
            anomaly_ratio=args.anomaly_ratio,
            seed=seed,
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
        "photometric_history_min": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_min"),
        "photometric_history_max": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_max"),
        "photometric_history_distance_threshold": cfg["models"][
            "dynamic_committee"
        ].get("photometric_history_distance_threshold"),
        "photometric_dino_similarity_floor": cfg["models"][
            "dynamic_committee"
        ].get("photometric_dino_similarity_floor"),
        "photometric_budget_replacement": bool(
            cfg["models"]["dynamic_committee"].get(
                "photometric_budget_replacement", True
            )
        ),
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
