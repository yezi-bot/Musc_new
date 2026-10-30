import math

import numpy as np
import torch
from sklearn import linear_model

from ._CHANNEL import ChannelMemory
from ._DYNAMIC_EXPERT import DynamicExpertManager
from ._MSM import (
    MSM2_online,
    aggregate_reference_distances,
    build_causal_fuser_training_data,
    fit_causal_detect_fuser,
    interval_average,
    safe_topmin_counts,
    score_dino_reference_committee,
)
from ._ROBUST_SHIFT import (
    pairwise_window_sinkhorn_profile,
    robust_sinkhorn_profile,
    robust_window_sinkhorn_profile,
    uniformly_sample_patch_ids,
)


def _controlled_contamination_patch_ids(
    patch_distances,
    patch_count,
    contamination_mode,
):
    patch_count = int(patch_count)
    if not 0 < patch_count <= int(patch_distances.numel()):
        raise ValueError("controlled patch count is out of range")
    ordered = torch.argsort(patch_distances)
    if contamination_mode == "patch":
        return ordered[-patch_count:].flip(0).tolist()
    if contamination_mode != "mixed":
        raise ValueError("patch selection requires patch or mixed mode")
    selected = []
    low = 0
    high = int(ordered.numel()) - 1
    for index in range(patch_count):
        if index % 2 == 0:
            selected.append(int(ordered[high]))
            high -= 1
        else:
            selected.append(int(ordered[low]))
            low += 1
    return selected


def _patch_to_history_image_distances(
    current,
    historical_features,
    reference_chunk_size=8,
):
    if not historical_features:
        return None
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be at least 1")

    patch_count, feature_dim = current.shape
    patch_to_image = []
    for start in range(0, len(historical_features), reference_chunk_size):
        references = torch.stack(
            historical_features[start : start + reference_chunk_size]
        ).to(current.device)
        distances = torch.cdist(
            current.unsqueeze(0),
            references.reshape(-1, feature_dim),
        ).reshape(
            patch_count,
            references.shape[0],
            patch_count,
        )
        patch_to_image.append(distances.amin(dim=-1))
    return torch.cat(patch_to_image, dim=1)


def _image_msm_from_distances(patch_to_image):
    patch_scores = interval_average(
        patch_to_image,
        topmin_min=0,
        topmin_max=0.3,
    )
    return float(patch_scores.max().cpu())


def _image_msm_score(current, historical_features, reference_chunk_size=8):
    distances = _patch_to_history_image_distances(
        current,
        historical_features,
        reference_chunk_size=reference_chunk_size,
    )
    return None if distances is None else _image_msm_from_distances(distances)


def _image_msm_profile(
    current,
    historical_features,
    short_windows=(8, 16, 32),
    reference_chunk_size=8,
):
    windows = tuple(int(window) for window in short_windows)
    if not windows or any(window < 1 for window in windows):
        raise ValueError("short_windows must contain positive integers")
    if len(set(windows)) != len(windows):
        raise ValueError("short_windows must not contain duplicates")

    distances = _patch_to_history_image_distances(
        current,
        historical_features,
        reference_chunk_size=reference_chunk_size,
    )
    if distances is None:
        return None, {str(window): None for window in windows}

    long_score = _image_msm_from_distances(distances)
    short_scores = {}
    for window in windows:
        short_scores[str(window)] = _image_msm_from_distances(
            distances[:, -min(window, distances.shape[1]) :]
        )
    return long_score, short_scores


def _channel_support(memory, features, grid_size, distance_threshold):
    distances = memory.patch_to_mature_distances(features, grid_size)
    if distances is None:
        return None, None
    if distance_threshold is None:
        return None, distances
    support = float((distances <= distance_threshold).float().mean())
    return support, distances


def _positioned_expert_support(
    current,
    reference,
    grid_size,
    distance_threshold,
    position_radius=1,
):
    if distance_threshold is None:
        return None
    if current.shape != reference.shape:
        raise ValueError("current and reference expert features must match")
    if isinstance(grid_size, int):
        rows = cols = int(grid_size)
    else:
        rows, cols = (int(grid_size[0]), int(grid_size[1]))
    if rows * cols != current.shape[0]:
        raise ValueError("grid_size does not match the patch count")
    patch_ids = torch.arange(current.shape[0], device=current.device)
    patch_rows = torch.div(patch_ids, cols, rounding_mode="floor")
    patch_cols = patch_ids % cols
    valid_positions = (
        (patch_rows[:, None] - patch_rows[None, :]).abs() <= position_radius
    ) & (
        (patch_cols[:, None] - patch_cols[None, :]).abs() <= position_radius
    )
    distances = torch.cdist(current, reference)
    distances = distances.masked_fill(~valid_positions, float("inf"))
    nearest = distances.amin(dim=1)
    return float((nearest <= distance_threshold).float().mean())


def _robust_shift_audit_profile(
    current,
    current_step,
    active_experts,
    committee_features,
    grid_size,
    max_patches,
    epsilon,
    iterations,
):
    if not active_experts:
        return {
            "nearest_expert_id": None,
            "nearest_discrete_cost": None,
            "expert_profiles": {},
        }

    patch_ids = uniformly_sample_patch_ids(
        current.shape[0],
        max_patches,
        current.device,
    )
    target = current[patch_ids]
    profiles = {}
    for expert in active_experts:
        reference_image_id = int(expert["image_id"])
        if reference_image_id >= current_step:
            raise ValueError("robust shift audit requires historical experts")
        reference = committee_features[reference_image_id][patch_ids]
        profiles[str(expert["expert_id"])] = robust_sinkhorn_profile(
            target,
            reference,
            target_patch_ids=patch_ids,
            reference_patch_ids=patch_ids,
            grid_size=grid_size,
            position_radius=None,
            epsilon=epsilon,
            iterations=iterations,
        )

    nearest_expert_id, nearest = min(
        profiles.items(),
        key=lambda item: item[1]["discrete_cost"],
    )
    return {
        "nearest_expert_id": int(nearest_expert_id),
        "nearest_discrete_cost": nearest["discrete_cost"],
        "expert_profiles": profiles,
    }


def _robust_shift_window_audit_profile(
    current_step,
    selected_steps,
    window_steps,
    active_experts,
    committee_features,
    grid_size,
    max_patches,
    position_radius,
    epsilon,
    iterations,
    fixed_reference_expert=None,
):
    selected_steps = [int(step) for step in selected_steps]
    window_steps = [int(step) for step in window_steps]
    if len(selected_steps) < 2:
        return {
            "state": "insufficient_window",
            "window_steps": window_steps,
            "selected_steps": selected_steps,
            "dropped_steps": [
                step for step in window_steps if step not in selected_steps
            ],
            "nearest_expert_id": None,
            "nearest_discrete_cost": None,
            "fixed_reference_expert_id": (
                int(fixed_reference_expert["expert_id"])
                if fixed_reference_expert is not None
                else None
            ),
            "fixed_reference_image_id": (
                int(fixed_reference_expert["image_id"])
                if fixed_reference_expert is not None
                else None
            ),
            "fixed_reference_discrete_cost": None,
            "fixed_reference_profile": None,
            "within_window_discrete_cost": None,
            "within_window_discrete_cost_median": None,
            "within_window_discrete_cost_max": None,
            "within_window_sinkhorn_cost_median": None,
            "within_window_sinkhorn_cost_max": None,
            "within_window_mutual_fraction": None,
            "within_window_mutual_fraction_mean": None,
            "within_window_mutual_fraction_min": None,
            "within_window_pair_count": 0,
            "within_window_pair_profiles": {},
            "expert_profiles": {},
        }
    if any(step > current_step for step in selected_steps):
        raise ValueError("robust shift window cannot contain future steps")

    patch_ids = uniformly_sample_patch_ids(
        committee_features.shape[1],
        max_patches,
        committee_features.device,
    )
    target_window = committee_features[selected_steps]
    within_profile = pairwise_window_sinkhorn_profile(
        target_window,
        patch_ids=patch_ids,
        grid_size=grid_size,
        position_radius=position_radius,
        epsilon=epsilon,
        iterations=iterations,
    )
    within_pair_profiles = {}
    for pair_profile in within_profile["pair_profiles"]:
        left_step = selected_steps[pair_profile["left_image_index"]]
        right_step = selected_steps[pair_profile["right_image_index"]]
        pair_key = f"{min(left_step, right_step)}:{max(left_step, right_step)}"
        pair_values = {
            key: value
            for key, value in pair_profile.items()
            if key not in {"left_image_index", "right_image_index"}
        }
        within_pair_profiles[pair_key] = {
            "target_step": left_step,
            "reference_step": right_step,
            **pair_values,
        }

    profiles = {}
    for expert in active_experts:
        reference_image_id = int(expert["image_id"])
        if reference_image_id >= current_step:
            raise ValueError("robust shift audit requires historical experts")
        profiles[str(expert["expert_id"])] = (
            robust_window_sinkhorn_profile(
                target_window,
                committee_features[reference_image_id].unsqueeze(0),
                patch_ids=patch_ids,
                grid_size=grid_size,
                position_radius=position_radius,
                epsilon=epsilon,
                iterations=iterations,
            )
        )

    nearest_expert_id = None
    nearest_discrete_cost = None
    if profiles:
        nearest_expert_id, nearest = min(
            profiles.items(),
            key=lambda item: item[1]["discrete_cost"],
        )
        nearest_expert_id = int(nearest_expert_id)
        nearest_discrete_cost = nearest["discrete_cost"]

    fixed_reference_expert_id = None
    fixed_reference_image_id = None
    fixed_reference_profile = None
    if fixed_reference_expert is not None:
        fixed_reference_expert_id = int(
            fixed_reference_expert["expert_id"]
        )
        fixed_reference_image_id = int(
            fixed_reference_expert["image_id"]
        )
        if fixed_reference_image_id >= current_step:
            raise ValueError(
                "fixed robust shift reference must be historical"
            )
        fixed_reference_profile = robust_window_sinkhorn_profile(
            target_window,
            committee_features[
                fixed_reference_image_id
            ].unsqueeze(0),
            patch_ids=patch_ids,
            grid_size=grid_size,
            position_radius=position_radius,
            epsilon=epsilon,
            iterations=iterations,
        )

    return {
        "state": (
            "observed"
            if profiles or fixed_reference_profile is not None
            else "no_active_expert"
        ),
        "window_steps": window_steps,
        "selected_steps": selected_steps,
        "dropped_steps": [
            step for step in window_steps if step not in selected_steps
        ],
        "nearest_expert_id": nearest_expert_id,
        "nearest_discrete_cost": nearest_discrete_cost,
        "fixed_reference_expert_id": fixed_reference_expert_id,
        "fixed_reference_image_id": fixed_reference_image_id,
        "fixed_reference_discrete_cost": (
            fixed_reference_profile["discrete_cost"]
            if fixed_reference_profile is not None
            else None
        ),
        "fixed_reference_profile": fixed_reference_profile,
        "within_window_discrete_cost": (
            within_profile["discrete_cost_median"]
        ),
        "within_window_discrete_cost_median": (
            within_profile["discrete_cost_median"]
        ),
        "within_window_discrete_cost_max": (
            within_profile["discrete_cost_max"]
        ),
        "within_window_sinkhorn_cost_median": (
            within_profile["sinkhorn_cost_median"]
        ),
        "within_window_sinkhorn_cost_max": (
            within_profile["sinkhorn_cost_max"]
        ),
        "within_window_mutual_fraction": (
            within_profile["mutual_fraction_mean"]
        ),
        "within_window_mutual_fraction_mean": (
            within_profile["mutual_fraction_mean"]
        ),
        "within_window_mutual_fraction_min": (
            within_profile["mutual_fraction_min"]
        ),
        "within_window_pair_count": within_profile["pair_count"],
        "within_window_pair_profiles": within_pair_profiles,
        "expert_profiles": profiles,
    }


def build_dynamic_committee_timeline(
    committee_features,
    device,
    lab_features=None,
    position_radius=1,
    channel_distance_quantile=0.7,
    reliability_alpha=0.5,
    channel_ttl=5,
    mature_span=3.0,
    density_k=5,
    memory_reliability_threshold=0.5,
    ms_short_windows=(8, 16, 32),
    robust_shift_audit=False,
    robust_shift_max_patches=128,
    robust_shift_epsilon=0.05,
    robust_shift_iterations=20,
    robust_shift_window_size=4,
    robust_shift_window_keep=3,
    robust_shift_position_radius=1,
    manager_kwargs=None,
    reset_steps=None,
    contamination_injections=None,
    controlled_history_evictions=None,
):
    if committee_features.ndim != 3:
        raise ValueError("committee_features must have shape [image, patch, feature]")
    if lab_features is not None:
        if not isinstance(lab_features, torch.Tensor):
            lab_features = torch.as_tensor(lab_features)
        if lab_features.shape != (committee_features.shape[0], 6):
            raise ValueError("lab_features must have shape [image, 6]")
        if not torch.isfinite(lab_features).all():
            raise ValueError("lab_features must be finite")
    if not 0.0 <= memory_reliability_threshold <= 1.0:
        raise ValueError(
            "memory_reliability_threshold must be within [0, 1]"
        )
    if robust_shift_max_patches < 1:
        raise ValueError("robust_shift_max_patches must be positive")
    if robust_shift_epsilon <= 0.0:
        raise ValueError("robust_shift_epsilon must be positive")
    if robust_shift_iterations < 1:
        raise ValueError("robust_shift_iterations must be positive")
    if robust_shift_window_size < 1:
        raise ValueError("robust_shift_window_size must be positive")
    if not 2 <= robust_shift_window_keep <= robust_shift_window_size:
        raise ValueError(
            "robust_shift_window_keep must be between 2 and the window size"
        )
    if robust_shift_position_radius is not None and (
        robust_shift_position_radius < 0
    ):
        raise ValueError("robust_shift_position_radius must be non-negative")
    ms_short_windows = tuple(int(window) for window in ms_short_windows)
    grid_size = math.isqrt(committee_features.shape[1])
    if grid_size * grid_size != committee_features.shape[1]:
        raise ValueError("committee patch count must form a square grid")

    reset_steps = set(reset_steps or [])
    contamination_injections = {
        int(step): dict(spec)
        for step, spec in (contamination_injections or {}).items()
    }
    controlled_history_evictions = {
        int(step): dict(spec)
        for step, spec in (controlled_history_evictions or {}).items()
    }
    invalid_contamination_steps = [
        step
        for step in contamination_injections
        if step < 0 or step >= committee_features.shape[0]
    ]
    if invalid_contamination_steps:
        raise ValueError(
            "contamination injection steps must be inside the feature stream"
        )
    invalid_eviction_steps = [
        step
        for step in controlled_history_evictions
        if step < 0 or step >= committee_features.shape[0] - 1
    ]
    if invalid_eviction_steps:
        raise ValueError(
            "controlled history eviction steps must leave a future sample"
        )
    invalid_reset_steps = [
        step
        for step in reset_steps
        if step <= 0 or step >= committee_features.shape[0]
    ]
    if invalid_reset_steps:
        raise ValueError("reset_steps must be inside the feature stream")

    def new_memory():
        return ChannelMemory(
            max_ttl=channel_ttl,
            mature_span=mature_span,
            density_k=density_k,
            device=device,
            position_radius=position_radius,
        )

    memory = new_memory()
    manager = DynamicExpertManager(**(manager_kwargs or {}))
    historical_features = []
    distance_history = []
    timeline = []
    segment_start = 0
    robust_window_members = []
    fixed_reference_expert = None
    controlled_clean_expert_ids = set()
    controlled_replay_specs = {}

    for step in range(committee_features.shape[0]):
        state_reset_before = step in reset_steps
        if state_reset_before:
            next_expert_id = manager.next_expert_id
            memory = new_memory()
            manager = DynamicExpertManager(**(manager_kwargs or {}))
            manager.next_expert_id = next_expert_id
            manager.last_step = step - 1
            historical_features = []
            distance_history = []
            robust_window_members = []
            fixed_reference_expert = None
            controlled_clean_expert_ids = set()
            segment_start = step
        for expert in manager.active_experts:
            if int(expert["expert_id"]) in controlled_clean_expert_ids:
                expert["scoring_weight"] = 1.0
                expert["member_reliability"] = 1.0
                if expert.get("scoring_patch_reliabilities") is not None:
                    expert["scoring_patch_reliabilities"] = [
                        1.0
                        for _ in expert["scoring_patch_reliabilities"]
                    ]
        current = committee_features[step]
        controlled_replay = controlled_replay_specs.get(step)
        if controlled_replay is not None:
            source_image_id = int(
                controlled_replay["source_image_id"]
            )
            clean_image_id = int(controlled_replay["clean_image_id"])
            if source_image_id >= step or clean_image_id >= step:
                raise ValueError(
                    "controlled replay references must be historical"
                )
            replay_patch_ids = torch.as_tensor(
                controlled_replay["patch_ids"],
                device=committee_features.device,
                dtype=torch.long,
            )
            current = committee_features[clean_image_id].clone()
            current.index_copy_(
                0,
                replay_patch_ids,
                committee_features[source_image_id].index_select(
                    0,
                    replay_patch_ids,
                ),
            )
        current_lab = (
            lab_features[step]
            if lab_features is not None
            else None
        )
        active_before = manager.active_experts_before_step()
        expert_slots_before = manager.expert_slots_before_step()
        diversity_audit_before = manager.diversity_audit_before_step()
        # 读取当前候选池状态
        provisional_before = manager.provisional_expert_before_step()
        training_members = manager.fuser_training_members(step)
        ms_score, ms_short_scores = _image_msm_profile(
            current,
            historical_features,
            short_windows=ms_short_windows,
        )
        primary_short_window = int(ms_short_windows[0])
        ms_short = ms_short_scores[str(primary_short_window)]
        ms_short_ratio = (
            ms_short / ms_score
            if ms_score is not None and ms_score > 0.0
            else None
        )
        distance_threshold = (
            float(np.quantile(distance_history, channel_distance_quantile))
            if distance_history
            else None
        )
        channel_support, current_distances = _channel_support(
            memory,
            current,
            grid_size,
            distance_threshold,
        )
        association_profile = memory.association_profile(
            current,
            grid_size,
            current_image_id=step,
        )
        channel_ids, _ = memory.patch_associations(current, grid_size)
        _, reliability = memory.patch_reliability(current, grid_size)
        soft_reliability = (
            reliability_alpha + (1.0 - reliability_alpha) * reliability
        )
        memory_patch_ids = torch.nonzero(
            (channel_ids >= 0)
            & (reliability >= memory_reliability_threshold),
            as_tuple=False,
        ).flatten().tolist()

        expert_supports = {}
        current_expert_supports = {}
        for expert in expert_slots_before:
            expert_features = committee_features[expert["image_id"]]
            support, _ = _channel_support(
                memory,
                expert_features,
                grid_size,
                distance_threshold,
            )
            expert_supports[expert["expert_id"]] = support
            current_expert_supports[expert["expert_id"]] = (
                _positioned_expert_support(
                    current,
                    expert_features,
                    grid_size,
                    distance_threshold,
                )
            )

        robust_shift_profile = None
        if robust_shift_audit:
            selection_score = (
                ms_short if ms_short is not None else ms_score
            )
            robust_window_members.append(
                {
                    "step": int(step),
                    "selection_score": selection_score,
                }
            )
            del robust_window_members[:-robust_shift_window_size]
            window_steps = [
                member["step"] for member in robust_window_members
            ]
            ranked_members = sorted(
                robust_window_members,
                key=lambda member: (
                    member["selection_score"] is None,
                    member["selection_score"]
                    if member["selection_score"] is not None
                    else float("inf"),
                ),
            )
            selected_steps = [
                member["step"]
                for member in ranked_members[:robust_shift_window_keep]
            ]
            if len(robust_window_members) < robust_shift_window_keep:
                selected_steps = []
            robust_shift_profile = _robust_shift_window_audit_profile(
                step,
                selected_steps,
                window_steps,
                active_before,
                committee_features,
                grid_size,
                robust_shift_max_patches,
                robust_shift_position_radius,
                robust_shift_epsilon,
                robust_shift_iterations,
                fixed_reference_expert=fixed_reference_expert,
            )

        lab_drift_profile = manager.advance_lab_state(
            current_lab,
            step=step,
            image_id=step,
        )
        drift_candidate_audit = manager.advance_drift_candidate_audit(
            step=step,
            image_id=step,
            dino_patch_features=current,
            lab_feature=current_lab,
            ms_short_score=ms_short,
            lab_drift_profile=lab_drift_profile,
            channel_support=channel_support,
            mature_match_fraction=association_profile[
                "mature_match_fraction"
            ],
            recent_provisional_match_fraction=association_profile[
                "recent_provisional_match_fraction"
            ],
        )

        event = manager.advance(
            step=step,
            image_id=step,
            dino_patch_features=current,
            ms_score=ms_score,
            channel_support=channel_support,
            expert_channel_supports=expert_supports,
            current_expert_supports=current_expert_supports,
            provisional_channel_support=association_profile[
                "recent_provisional_match_fraction"
            ],
            ms_short_score=ms_short,
            ms_short_ratio=ms_short_ratio,
            memory_patch_ids=memory_patch_ids,
            patch_reliabilities=soft_reliability,
            lab_feature=current_lab,
            lab_drift_profile=lab_drift_profile,
        )

        if (
            robust_shift_audit
            and fixed_reference_expert is None
            and event["admitted_expert_id"] is not None
        ):
            admitted_expert_id = int(event["admitted_expert_id"])
            fixed_reference_expert = next(
                {
                    "expert_id": int(expert["expert_id"]),
                    "image_id": int(expert["image_id"]),
                    "admission_step": int(step),
                }
                for expert in manager.active_experts
                if int(expert["expert_id"]) == admitted_expert_id
            )

        if current_distances is not None:
            finite = current_distances[torch.isfinite(current_distances)]
            distance_history.extend(finite.tolist())

        memory.update(
            current,
            image_id=step,
            grid_size=grid_size,
            patch_reliabilities=soft_reliability,
        )
        controlled_history_eviction = None
        eviction = controlled_history_evictions.get(step)
        if eviction is not None:
            target_expert_id = int(eviction.get("expert_id", 0))
            victim = next(
                (
                    expert
                    for expert in manager.active_experts
                    if int(expert["expert_id"]) == target_expert_id
                ),
                None,
            )
            victim_snapshot = (
                {
                    "expert_id": int(victim["expert_id"]),
                    "image_id": int(victim["image_id"]),
                    "cluster_id": int(victim["cluster_id"]),
                    "admission_step": int(victim["admission_step"]),
                }
                if victim is not None
                else None
            )
            evicted = manager._replace_budgeted_expert(
                target_expert_id,
                step,
                {
                    "replacement_reason": "controlled_history_eviction",
                    "controlled_evaluation": True,
                },
            )
            controlled_history_eviction = {
                "attempted": True,
                "evicted": bool(evicted),
                "step": int(step),
                "target_expert_id": target_expert_id,
                "victim": victim_snapshot,
                "reason": (
                    "controlled_history_eviction"
                    if evicted
                    else "target_not_active"
                ),
            }
        controlled_contamination = None
        injection = contamination_injections.get(step)
        if injection is not None:
            step_start_experts = {
                int(expert["expert_id"]): expert for expert in active_before
            }
            for expert in manager.active_experts:
                snapshot = step_start_experts.get(int(expert["expert_id"]))
                if snapshot is None:
                    continue
                expert["image_id"] = int(snapshot["image_id"])
                expert["scoring_weight"] = 1.0
                expert["member_reliability"] = 1.0
                expert["global_member_reliability"] = 1.0
                if snapshot.get("scoring_patch_reliabilities") is not None:
                    expert["scoring_patch_reliabilities"] = [
                        1.0
                        for _ in snapshot["scoring_patch_reliabilities"]
                    ]
                    expert["local_patch_reliabilities"] = list(
                        expert["scoring_patch_reliabilities"]
                    )
                expert.pop("controlled_contamination_mode", None)
                expert.pop("controlled_clean_image_id", None)
                expert.pop("controlled_contaminated_patch_ids", None)
            clean_expert_ids = [
                expert_id
                for expert_id in step_start_experts
                if any(
                    int(expert["expert_id"]) == expert_id
                    for expert in manager.active_experts
                )
            ]
            contamination_mode = str(injection.get("mode", "member"))
            if contamination_mode not in {"member", "patch", "mixed"}:
                raise ValueError(
                    "controlled contamination mode must be member, patch, or mixed"
                )
            injected_features = current
            injected_patch_reliabilities = soft_reliability
            contaminated_patch_ids = list(range(current.shape[0]))
            if contamination_mode in {"patch", "mixed"}:
                if not active_before:
                    raise ValueError(
                        "patch contamination requires a clean active expert"
                    )
                clean_image_id = int(active_before[0]["image_id"])
                clean_features = committee_features[clean_image_id]
                patch_ratio = float(injection.get("patch_ratio", 0.3))
                if not 0.0 < patch_ratio <= 1.0:
                    raise ValueError(
                        "controlled contamination patch_ratio must be in (0, 1]"
                    )
                contaminated_patch_count = max(
                    1,
                    int(round(current.shape[0] * patch_ratio)),
                )
                patch_distances = torch.linalg.vector_norm(
                    current - clean_features,
                    dim=1,
                )
                contaminated_patch_ids = (
                    _controlled_contamination_patch_ids(
                        patch_distances,
                        contaminated_patch_count,
                        contamination_mode,
                    )
                )
                injected_features = clean_features.clone()
                injected_features[contaminated_patch_ids] = current[
                    contaminated_patch_ids
                ]
                clean_patch_reliabilities = active_before[0].get(
                    "scoring_patch_reliabilities"
                )
                if clean_patch_reliabilities is None:
                    clean_patch_reliabilities = torch.ones_like(
                        soft_reliability
                    )
                injected_patch_reliabilities = torch.as_tensor(
                    clean_patch_reliabilities,
                    device=soft_reliability.device,
                    dtype=soft_reliability.dtype,
                ).clone()
                contaminated_patch_tensor = torch.as_tensor(
                    contaminated_patch_ids,
                    device=soft_reliability.device,
                    dtype=torch.long,
                )
                injected_patch_reliabilities.index_copy_(
                    0,
                    contaminated_patch_tensor,
                    soft_reliability.index_select(
                        0,
                        contaminated_patch_tensor,
                    ),
                )
            member = manager._quarantine_member(
                step=step,
                image_id=step,
                dino_patch_features=injected_features,
                ms_score=ms_score,
                ms_short_score=ms_short,
                ms_short_ratio=ms_short_ratio,
                channel_support=channel_support,
                provisional_channel_support=association_profile[
                    "recent_provisional_match_fraction"
                ],
                memory_patch_ids=memory_patch_ids,
                lab_feature=current_lab,
                candidate_route="controlled_contamination",
                historical_ms_threshold=event.get("ms_threshold"),
                current_expert_supports=current_expert_supports,
                patch_reliabilities=injected_patch_reliabilities,
            )
            reliability_context_count = 0
            if manager.candidate_reliability_enabled and active_before:
                reliability_context = [member]
                for expert in active_before:
                    expert_image_id = int(expert["image_id"])
                    context_patch_reliabilities = expert.get(
                        "scoring_patch_reliabilities"
                    )
                    reliability_context.append(
                        manager._quarantine_member(
                            step=step,
                            image_id=expert_image_id,
                            dino_patch_features=committee_features[
                                expert_image_id
                            ],
                            ms_score=None,
                            ms_short_score=None,
                            ms_short_ratio=None,
                            channel_support=expert_supports.get(
                                int(expert["expert_id"])
                            ),
                            provisional_channel_support=expert_supports.get(
                                int(expert["expert_id"])
                            ),
                            memory_patch_ids=None,
                            lab_feature=None,
                            candidate_route=(
                                "controlled_contamination_context"
                            ),
                            current_expert_supports=None,
                            patch_reliabilities=(
                                context_patch_reliabilities
                                if context_patch_reliabilities is not None
                                else torch.ones_like(soft_reliability)
                            ),
                        )
                    )
                manager._refresh_candidate_reliabilities(
                    reliability_context
                )
                reliability_context_count = len(reliability_context) - 1
                if (
                    contamination_mode == "patch"
                    and manager.candidate_patch_reliability_mode
                    == "hierarchical_global_local"
                ):
                    clean_global = float(
                        active_before[0].get(
                            "global_member_reliability",
                            active_before[0].get(
                                "member_reliability",
                                1.0,
                            ),
                        )
                    )
                    member["global_member_reliability"] = clean_global
                    member["member_reliability"] = clean_global
                    base = torch.as_tensor(
                        member["base_patch_reliabilities"],
                        dtype=torch.float32,
                    )
                    member["patch_reliabilities"] = (
                        base * clean_global
                    ).clamp(
                        min=manager.candidate_reliability_floor,
                        max=1.0,
                    )
                    patch_bank = member.get("reliability_patch_bank")
                    if patch_bank is not None:
                        member["patch_prototypes"] = manager._patch_coreset(
                            patch_bank["features"],
                            patch_reliabilities=member[
                                "patch_reliabilities"
                            ],
                        )
                    member.setdefault("reliability_audit", {})[
                        "controlled_global_context"
                    ] = "step_start_clean_expert"
                    member["reliability_audit"][
                        "global_member_reliability"
                    ] = clean_global
            cluster_id, injected_expert_id = manager._admit_quarantined_member(
                member,
                step=step,
                cluster_members=[member],
                admission_route="controlled_contamination",
                support_threshold=event.get("support_threshold"),
                candidate_evidence={
                    "controlled_contamination": True,
                },
                preserve_candidate_reliability=(
                    manager.candidate_reliability_enabled
                    and reliability_context_count > 0
                ),
            )
            if (
                injected_expert_id is not None
                and contamination_mode in {"patch", "mixed"}
            ):
                injected_expert = next(
                    expert
                    for expert in manager.active_experts
                    if int(expert["expert_id"]) == int(injected_expert_id)
                )
                injected_expert["controlled_contamination_mode"] = (
                    contamination_mode
                )
                injected_expert["controlled_clean_image_id"] = int(
                    active_before[0]["image_id"]
                )
                injected_expert[
                    "controlled_contaminated_patch_ids"
                ] = list(contaminated_patch_ids)
                for replay_step in injection.get("replay_steps", []):
                    replay_step = int(replay_step)
                    if replay_step <= step:
                        raise ValueError(
                            "controlled replay must occur after injection"
                        )
                    controlled_replay_specs[replay_step] = {
                        "source_image_id": int(step),
                        "clean_image_id": int(active_before[0]["image_id"]),
                        "patch_ids": list(contaminated_patch_ids),
                    }
            if injection.get("promote_clean_experts", True):
                controlled_clean_expert_ids.update(clean_expert_ids)
                for expert in manager.active_experts:
                    if int(expert["expert_id"]) in controlled_clean_expert_ids:
                        expert["scoring_weight"] = 1.0
                        expert["member_reliability"] = 1.0
                        expert["global_member_reliability"] = 1.0
                        if (
                            expert.get("scoring_patch_reliabilities")
                            is not None
                        ):
                            expert["scoring_patch_reliabilities"] = [
                                1.0
                                for _ in expert[
                                    "scoring_patch_reliabilities"
                                ]
                            ]
                            expert["local_patch_reliabilities"] = list(
                                expert["scoring_patch_reliabilities"]
                            )
            controlled_contamination = {
                "injected": injected_expert_id is not None,
                "source_image_id": int(step),
                "source_label": injection.get("source_label", "anomaly"),
                "source_path": injection.get("source_path"),
                "injected_step": int(step),
                "injected_cluster_id": (
                    int(cluster_id) if cluster_id is not None else None
                ),
                "injected_expert_id": (
                    int(injected_expert_id)
                    if injected_expert_id is not None
                    else None
                ),
                "admission_route": "controlled_contamination",
                "contamination_mode": contamination_mode,
                "patch_selection_policy": (
                    "largest_distance"
                    if contamination_mode == "patch"
                    else "alternating_distance_extremes"
                ),
                "base_patch_reliability_source": (
                    "feature_provenance_hybrid"
                    if contamination_mode in {"patch", "mixed"}
                    else "source_member"
                ),
                "base_patch_reliability_mean": float(
                    injected_patch_reliabilities.float().mean()
                ),
                "base_patch_reliability_min": float(
                    injected_patch_reliabilities.float().min()
                ),
                "contaminated_patch_ids": contaminated_patch_ids,
                "contaminated_patch_count": len(contaminated_patch_ids),
                "contaminated_patch_ratio": (
                    len(contaminated_patch_ids) / current.shape[0]
                ),
                "replay_steps": [
                    int(replay_step)
                    for replay_step in injection.get("replay_steps", [])
                ],
                "controlled_clean_expert_ids": sorted(
                    controlled_clean_expert_ids
                ),
                "reliability_context_count": reliability_context_count,
                "member_reliability": (
                    float(member.get("member_reliability", 1.0))
                ),
                "scoring_patch_reliability_mean": (
                    float(
                        torch.as_tensor(
                            member.get(
                                "scoring_patch_reliabilities",
                                torch.ones_like(soft_reliability),
                            )
                        ).float().mean()
                    )
                ),
                "scoring_patch_reliability_min": (
                    float(
                        torch.as_tensor(
                            member.get(
                                "scoring_patch_reliabilities",
                                torch.ones_like(soft_reliability),
                            )
                        ).float().min()
                    )
                ),
            }
        historical_features.append(current.detach().cpu())

        timeline.append(
            {
                "step": step,
                "segment_start": segment_start,
                "state_reset_before": state_reset_before,
                "active_experts_before": active_before,
                "expert_slots_before": expert_slots_before,
                "diversity_audit_before": diversity_audit_before,
                "provisional_expert_before": provisional_before,
                "fuser_training_members_before": training_members,
                "ms_score": ms_score,
                "ms_long": ms_score,
                "ms_short": ms_short,
                "ms_short_ratio": ms_short_ratio,
                "ms_short_window": primary_short_window,
                "ms_short_scores": ms_short_scores,
                "ms_short_history_sizes": {
                    str(window): min(int(window), len(historical_features))
                    for window in ms_short_windows
                },
                "channel_distance_threshold": distance_threshold,
                "channel_support": channel_support,
                "association_profile": association_profile,
                "memory_patch_candidate_count": len(memory_patch_ids),
                "memory_reliability_threshold": memory_reliability_threshold,
                "candidate_route": event["candidate_route"],
                "admission_route": event["admission_route"],
                "support_threshold": event["support_threshold"],
                "novel_quarantine": event["novel_quarantine"],
                "photometric_gate": event["photometric_gate"],
                "budget_replacement": event["budget_replacement"],
                "budget_replacement_evaluation": event[
                    "budget_replacement_evaluation"
                ],
                "history_reentry": event.get("history_reentry"),
                "fitness_history": event.get("fitness_history"),
                "historical_states": event.get("historical_states", []),
                "state_lifecycle_events": event.get(
                    "state_lifecycle_events", []
                ),
                "bootstrap": event["bootstrap"],
                "lab_drift_profile": lab_drift_profile,
                "drift_candidate_audit": drift_candidate_audit,
                "controlled_contamination": controlled_contamination,
                "controlled_history_eviction": (
                    controlled_history_eviction
                ),
                "controlled_contamination_replay": (
                    dict(controlled_replay)
                    if controlled_replay is not None
                    else None
                ),
                "novel_support_threshold": event[
                    "novel_support_threshold"
                ],
                "novel_ms_threshold": event["novel_ms_threshold"],

                "expert_channel_supports": expert_supports,
                "current_expert_supports": current_expert_supports,
                "robust_shift_profile": robust_shift_profile,
                "fixed_reference_expert_after": (
                    dict(fixed_reference_expert)
                    if fixed_reference_expert is not None
                    else None
                ),
                "admitted_expert_id": event["admitted_expert_id"],
                "deleted_expert_ids": event["deleted_expert_ids"],
                "expert_lifecycle_events": event[
                    "expert_lifecycle_events"
                ],
                "active_experts_after": manager.active_experts_before_step(),
                "expert_slots_after": manager.expert_slots_before_step(),
                "active_expert_ids_after": event["active_expert_ids_after_step"],
                "expert_slot_ids_after": event["expert_slot_ids_after_step"],
                "channel_count_after": len(memory.channels),
                "mature_channel_count_after": len(memory.mature_channels()),
            }
        )
    return timeline


def _strict_history_fallback(
    current,
    history,
    reason,
    reference_chunk_size=8,
):
    if history.shape[0] == 0:
        return torch.zeros(
            current.shape[0],
            device=current.device,
            dtype=torch.float32,
        ), "first_image_unavailable"
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be at least 1")
    patch_count, feature_dim = current.shape
    patch_to_image = []
    for start in range(0, history.shape[0], reference_chunk_size):
        references = history[start : start + reference_chunk_size]
        distances = torch.cdist(
            current.unsqueeze(0),
            references.reshape(-1, feature_dim),
        ).reshape(
            patch_count,
            references.shape[0],
            references.shape[1],
        )
        patch_to_image.append(distances.amin(dim=-1))
    return interval_average(
        torch.cat(patch_to_image, dim=1),
        topmin_min=0,
        topmin_max=0.3,
    ), reason


def _strict_history_dino_fallback(current_dino, dino_history):
    return _strict_history_fallback(
        current_dino,
        dino_history,
        "strict_history_dino",
    )


def _strict_history_clip_fallback(current_clip, clip_history):
    return _strict_history_fallback(
        current_clip,
        clip_history,
        "strict_history_clip",
    )


def _positioned_memory_scores(
    query,
    member_features,
    member_patch_ids,
    grid_size,
    position_radius=1,
    topmin_min=0.02,
    topmin_max=0.3,
):
    if query.ndim != 2:
        raise ValueError("query must have shape [patch_count, feature_dim]")
    if grid_size * grid_size != query.shape[0]:
        raise ValueError("query patch count must match grid_size")
    if len(member_features) != len(member_patch_ids):
        raise ValueError("member feature and patch-id counts must match")
    if position_radius < 0:
        raise ValueError("position_radius must be non-negative")

    patch_count = query.shape[0]
    query_ids = torch.arange(patch_count, device=query.device)
    query_rows = query_ids // grid_size
    query_cols = query_ids % grid_size
    per_member = []
    for features, patch_ids in zip(member_features, member_patch_ids):
        patch_ids = torch.as_tensor(
            patch_ids,
            device=query.device,
            dtype=torch.long,
        )
        if patch_ids.numel() == 0:
            continue
        references = features.to(query.device)[patch_ids]
        reference_rows = patch_ids // grid_size
        reference_cols = patch_ids % grid_size
        valid = (
            (query_rows[:, None] - reference_rows[None, :]).abs()
            <= position_radius
        ) & (
            (query_cols[:, None] - reference_cols[None, :]).abs()
            <= position_radius
        )
        distances = torch.cdist(query.float(), references.float())
        nearest = distances.masked_fill(~valid, float("inf")).amin(dim=1)
        per_member.append(nearest)

    if not per_member:
        return torch.zeros(patch_count, device=query.device), torch.zeros(
            patch_count,
            device=query.device,
            dtype=torch.bool,
        )
    distances = torch.stack(per_member, dim=1)
    valid_counts = torch.isfinite(distances).sum(dim=1)
    scores = torch.zeros(patch_count, device=query.device)
    for reference_count in torch.unique(valid_counts).tolist():
        if reference_count == 0:
            continue
        mask = valid_counts == reference_count
        finite = distances[mask]
        finite = finite[torch.isfinite(finite)].reshape(-1, reference_count)
        scores[mask] = interval_average(
            finite,
            topmin_min=topmin_min,
            topmin_max=topmin_max,
        )
    return scores, valid_counts > 0


def _distance_audit(score_audit):
    result = {}
    dino_distance = score_audit.get("dino_distance")
    clip_distance = score_audit.get("clip_distance")
    for name, values in (("dino", dino_distance), ("clip", clip_distance)):
        if values is not None:
            result[f"{name}_min"] = float(values.min())
            result[f"{name}_max"] = float(values.max())
            result[f"{name}_mean"] = float(values.mean())
    if dino_distance is not None and clip_distance is not None:
        dino_std = dino_distance.float().std(unbiased=False)
        clip_std = clip_distance.float().std(unbiased=False)
        if dino_std > 0 and clip_std > 0:
            dino_z = (dino_distance - dino_distance.mean()) / dino_std
            clip_z = (clip_distance - clip_distance.mean()) / clip_std
            result["dino_clip_correlation"] = float(torch.mean(dino_z * clip_z))
            result["dino_clip_z_disagreement"] = float(
                torch.mean(torch.abs(dino_z - clip_z))
            )
        else:
            result["dino_clip_correlation"] = None
            result["dino_clip_z_disagreement"] = None
    reliability = score_audit.get("reference_reliability")
    if reliability is not None:
        score_shift = reliability["score_shift"]
        result.update(
            {
                "reference_reliability_mean": reliability[
                    "reference_reliability_mean"
                ],
                "reference_reliability_min": reliability[
                    "reference_reliability_min"
                ],
                "low_reliability_patch_fraction": reliability[
                    "low_reliability_fraction"
                ],
                "effective_reference_weight": reliability[
                    "effective_reference_weight"
                ],
                "raw_reference_weight": reliability[
                    "raw_reference_weight"
                ],
                "effective_reference_weight_ratio": float(
                    reliability["effective_reference_weight"]
                    / max(reliability["raw_reference_weight"], 1)
                ),
                "reliability_score_shift_mean": float(
                    score_shift.mean()
                ),
                "reliability_score_shift_max": float(
                    score_shift.max()
                ),
                "reliability_selection_changed_fraction": reliability[
                    "selection_changed_fraction"
                ],
                "selected_reference_raw_counts": reliability[
                    "selected_reference_raw_counts"
                ],
                "selected_reference_effective_weights": reliability[
                    "selected_reference_effective_weights"
                ],
                "reference_bank_reliability_mean": reliability[
                    "bank_reliability_mean"
                ],
                "reference_bank_reliability_min": reliability[
                    "bank_reliability_min"
                ],
            }
        )
    return result


def _reference_contributions(
    score_audit,
    expert_ids,
    member_image_ids=None,
    blend_weight=1.0,
):
    reliability = (
        score_audit.get("reference_reliability")
        if score_audit is not None
        else None
    )
    if reliability is None:
        return []
    raw_counts = reliability.get("selected_reference_raw_counts", [])
    effective_weights = reliability.get(
        "selected_reference_effective_weights", []
    )
    member_raw_counts = reliability.get(
        "selected_reference_member_raw_counts", {}
    )
    member_effective_weights = reliability.get(
        "selected_reference_member_effective_weights", {}
    )
    if not (
        len(raw_counts) == len(expert_ids)
        and len(effective_weights) == len(expert_ids)
    ):
        raise ValueError(
            "reference contribution audit must align with expert ids"
        )
    if member_image_ids is None:
        member_image_ids = reliability.get(
            "reference_member_ids", list(expert_ids)
        )
    if len(member_image_ids) != len(expert_ids):
        raise ValueError("member image ids must align with expert ids")
    return [
        {
            "expert_id": int(expert_id),
            "member_image_id": int(member_image_id),
            "selected_patch_count": int(
                member_raw_counts.get(str(member_image_id), 0)
            ),
            "member_is_contaminated": None,
            "raw_weight": float(raw) * float(blend_weight),
            "effective_weight": float(effective) * float(blend_weight),
            "member_raw_weight": float(
                member_raw_counts.get(str(member_image_id), 0)
            ) * float(blend_weight),
            "member_effective_weight": float(
                member_effective_weights.get(str(member_image_id), 0.0)
            ) * float(blend_weight),
            "blend_weight": float(blend_weight),
            "blend_contribution": float(effective) * float(blend_weight),
        }
        for expert_id, member_image_id, raw, effective in zip(
            expert_ids,
            member_image_ids,
            raw_counts,
            effective_weights,
        )
    ]


def _counterfactual_dino_score_without_references(
    current_dino,
    dino_features,
    active,
    excluded_image_ids,
    history_start,
    step,
    topmin_min,
    topmin_max,
    reliability_penalty,
):
    clean_indices = [
        index
        for index, expert in enumerate(active)
        if int(expert["image_id"]) not in excluded_image_ids
    ]
    if not clean_indices:
        return None
    clean_active = [active[index] for index in clean_indices]
    clean_image_ids = [int(expert["image_id"]) for expert in clean_active]
    clean_weights = [
        float(expert.get("scoring_weight", 1.0))
        for expert in clean_active
    ]
    clean_reliabilities = [
        expert.get("scoring_patch_reliabilities")
        for expert in clean_active
    ]

    def reliability_tensor(indices):
        if not any(clean_reliabilities[index] is not None for index in indices):
            return None
        return torch.stack(
            [
                torch.as_tensor(
                    clean_reliabilities[index],
                    device=current_dino.device,
                    dtype=current_dino.dtype,
                )
                if clean_reliabilities[index] is not None
                else torch.ones(
                    current_dino.shape[0],
                    device=current_dino.device,
                    dtype=current_dino.dtype,
                )
                for index in indices
            ]
        )

    full_indices = [
        index for index, weight in enumerate(clean_weights) if weight >= 1.0
    ]
    if full_indices:
        full_image_ids = [clean_image_ids[index] for index in full_indices]
        counterfactual, _ = MSM2_online(
            current_dino,
            None,
            dino_features[full_image_ids],
            None,
            fusion_mode="dino_only",
            topmin_min=topmin_min,
            topmin_max=topmin_max,
            dino_patch_reliabilities=reliability_tensor(full_indices),
            dino_reference_member_ids=full_image_ids,
            reliability_penalty=reliability_penalty,
        )
    else:
        clean_history_ids = [
            image_id
            for image_id in range(history_start, step)
            if image_id not in excluded_image_ids
        ]
        if not clean_history_ids:
            return None
        counterfactual, _ = _strict_history_dino_fallback(
            current_dino,
            dino_features[clean_history_ids],
        )

    for index, weight in enumerate(clean_weights):
        if weight >= 1.0:
            continue
        confidence_score, _ = MSM2_online(
            current_dino,
            None,
            dino_features[[clean_image_ids[index]]],
            None,
            fusion_mode="dino_only",
            topmin_min=topmin_min,
            topmin_max=topmin_max,
            dino_patch_reliabilities=reliability_tensor([index]),
            dino_reference_member_ids=[clean_image_ids[index]],
            reliability_penalty=reliability_penalty,
        )
        counterfactual = counterfactual + weight * (
            confidence_score - counterfactual
        )
    return counterfactual


def _hierarchical_dino_score(
    current_dino,
    expert_dino,
    expert_patch_reliabilities,
    expert_global_reliabilities,
    expert_ids,
    expert_image_ids,
    history_score,
    *,
    topmin_min,
    topmin_max,
    reliability_penalty,
):
    expert_count = int(expert_dino.shape[0])
    if not (
        expert_count
        == len(expert_global_reliabilities)
        == len(expert_ids)
        == len(expert_image_ids)
    ):
        raise ValueError("hierarchical expert inputs must align")
    if expert_count < 1:
        raise ValueError("hierarchical scoring requires an active expert")
    global_reliabilities = torch.as_tensor(
        expert_global_reliabilities,
        device=current_dino.device,
        dtype=current_dino.dtype,
    )
    if not torch.isfinite(global_reliabilities).all():
        raise ValueError("global expert reliabilities contain non-finite values")
    if (
        (global_reliabilities < 0.0)
        | (global_reliabilities > 1.0)
    ).any():
        raise ValueError("global expert reliabilities must be within [0, 1]")
    if expert_patch_reliabilities is None:
        expert_patch_reliabilities = torch.ones(
            expert_dino.shape[:2],
            device=current_dino.device,
            dtype=current_dino.dtype,
        )

    if torch.equal(
        global_reliabilities,
        torch.ones_like(global_reliabilities),
    ):
        score, score_audit = MSM2_online(
            current_dino,
            None,
            expert_dino,
            None,
            fusion_mode="dino_only",
            topmin_min=topmin_min,
            topmin_max=topmin_max,
            dino_patch_reliabilities=expert_patch_reliabilities,
            dino_reference_member_ids=expert_image_ids,
            reliability_penalty=reliability_penalty,
            return_selection_details=True,
        )
        reliability = score_audit["reference_reliability"]
        raw_counts = reliability["selected_reference_raw_counts"]
        local_weights = reliability[
            "selected_reference_effective_weights"
        ]
        contributions = []
        for expert_id, image_id, raw, local in zip(
            expert_ids,
            expert_image_ids,
            raw_counts,
            local_weights,
        ):
            contributions.append(
                {
                    "expert_id": int(expert_id),
                    "member_image_id": int(image_id),
                    "selected_patch_count": int(raw),
                    "member_is_contaminated": None,
                    "raw_weight": float(raw),
                    "effective_weight": float(local),
                    "member_raw_weight": float(raw),
                    "member_effective_weight": float(local),
                    "blend_weight": 1.0,
                    "blend_contribution": float(local),
                    "raw_contribution": float(raw),
                    "local_effective_contribution": float(local),
                    "global_blend_contribution": float(local),
                }
            )
        score_audit["hierarchical_diagnostics"] = {
            "history_mean": float(history_score.mean()),
            "history_max": float(history_score.max()),
            "all_global_reliabilities_one": True,
        }
        return score, score_audit, contributions

    expert_scores = []
    local_expert_scores = []
    unweighted_expert_scores = []
    expert_selected_reliabilities = []
    for index in range(expert_count):
        unweighted_expert_score, _ = MSM2_online(
            current_dino,
            None,
            expert_dino[index : index + 1],
            None,
            fusion_mode="dino_only",
            topmin_min=topmin_min,
            topmin_max=topmin_max,
            dino_patch_reliabilities=torch.ones_like(
                expert_patch_reliabilities[index : index + 1]
            ),
            dino_reference_member_ids=[expert_image_ids[index]],
            reliability_penalty=reliability_penalty,
            return_selection_details=True,
        )
        expert_score, expert_audit = MSM2_online(
            current_dino,
            None,
            expert_dino[index : index + 1],
            None,
            fusion_mode="dino_only",
            topmin_min=topmin_min,
            topmin_max=topmin_max,
            dino_patch_reliabilities=(
                expert_patch_reliabilities[index : index + 1]
            ),
            dino_reference_member_ids=[expert_image_ids[index]],
            reliability_penalty=reliability_penalty,
            return_selection_details=True,
        )
        expert_scores.append(
            history_score
            + global_reliabilities[index]
            * (expert_score - history_score)
        )
        local_expert_scores.append(expert_score)
        unweighted_expert_scores.append(unweighted_expert_score)
        reliability = expert_audit["reference_reliability"]
        per_query = reliability[
            "contributing_reliabilities_per_query"
        ]
        expert_selected_reliabilities.append(
            torch.as_tensor(
                [values[0] for values in per_query],
                device=current_dino.device,
                dtype=current_dino.dtype,
            )
        )

    distances = torch.stack(expert_scores, dim=1)
    local_distances = torch.stack(local_expert_scores, dim=1)
    unweighted_distances = torch.stack(
        unweighted_expert_scores,
        dim=1,
    )
    k_min, k_max, _ = safe_topmin_counts(
        expert_count,
        topmin_min,
        topmin_max,
    )
    selected_experts = torch.topk(
        distances.float(),
        k_max,
        largest=False,
        sorted=True,
    ).indices[:, k_min:k_max]
    local_selected_experts = torch.topk(
        local_distances.float(),
        k_max,
        largest=False,
        sorted=True,
    ).indices[:, k_min:k_max]
    raw_selected_experts = torch.topk(
        unweighted_distances.float(),
        k_max,
        largest=False,
        sorted=True,
    ).indices[:, k_min:k_max]
    score = distances.gather(1, selected_experts).mean(dim=1)
    local_by_expert = torch.stack(
        expert_selected_reliabilities,
        dim=1,
    )
    selected_local = local_by_expert.gather(1, selected_experts)
    local_stage_reliabilities = local_by_expert.gather(
        1,
        local_selected_experts,
    )
    contributions = []
    for index, (expert_id, image_id) in enumerate(
        zip(expert_ids, expert_image_ids)
    ):
        global_selected_mask = selected_experts == index
        local_selected_mask = local_selected_experts == index
        raw_selected_mask = raw_selected_experts == index
        raw = int(raw_selected_mask.sum())
        local = float(
            local_stage_reliabilities[local_selected_mask].sum()
        )
        global_contribution = float(global_reliabilities[index]) * float(
            selected_local[global_selected_mask].sum()
        )
        contributions.append(
            {
                "expert_id": int(expert_id),
                "member_image_id": int(image_id),
                "selected_patch_count": raw,
                "member_is_contaminated": None,
                "raw_weight": float(raw),
                "effective_weight": local,
                "member_raw_weight": float(raw),
                "member_effective_weight": local,
                "blend_weight": float(global_reliabilities[index]),
                "blend_contribution": global_contribution,
                "raw_contribution": float(raw),
                "local_effective_contribution": local,
                "global_blend_contribution": global_contribution,
            }
        )
    score_audit = {
        "fusion_mode": "dino_only",
        "reference_count": expert_count,
        "dino_distance": score,
        "clip_distance": None,
        "fuser_score": None,
        "reference_reliability": None,
        "hierarchical_diagnostics": {
            "history_mean": float(history_score.mean()),
            "history_max": float(history_score.max()),
            "all_global_reliabilities_one": False,
            "experts": [
                {
                    "expert_id": int(expert_id),
                    "image_id": int(image_id),
                    "global_reliability": float(global_reliabilities[index]),
                    "raw_distance_mean": float(
                        unweighted_expert_scores[index].mean()
                    ),
                    "raw_distance_max": float(
                        unweighted_expert_scores[index].max()
                    ),
                    "local_distance_mean": float(
                        local_expert_scores[index].mean()
                    ),
                    "local_distance_max": float(
                        local_expert_scores[index].max()
                    ),
                    "blended_distance_mean": float(
                        expert_scores[index].mean()
                    ),
                    "blended_distance_max": float(
                        expert_scores[index].max()
                    ),
                    "raw_selected_patch_count": int(
                        (raw_selected_experts == index).sum()
                    ),
                    "local_selected_patch_count": int(
                        (local_selected_experts == index).sum()
                    ),
                    "global_selected_patch_count": int(
                        (selected_experts == index).sum()
                    ),
                }
                for index, (expert_id, image_id) in enumerate(
                    zip(expert_ids, expert_image_ids)
                )
            ],
        },
    }
    return score, score_audit, contributions


def score_dynamic_msm2_layer(
    dino_features,
    clip_features,
    timeline,
    fusion_mode="fuser",
    dino_weight=1.0,
    clip_weight=0.5,
    epsilon=1e-6,
    topmin_min=0.02,
    topmin_max=0.3,
    retrain_policy="on_change",
    training_source="cluster_history",
    committee_min_experts=3,
    committee_stable_steps=2,
    committee_change_ratio=0.4,
    committee_retrain_cooldown=5,
    position_radius=1,
    transition_fallback_window=8,
    transition_fallback_require_pre_pool_expert=False,
    transition_fallback_evidence_window=0,
    transition_fallback_min_hits=0,
    transition_fallback_latch=False,
    image_score_size=None,
    hierarchical_reliability_enabled=False,
):

    if transition_fallback_evidence_window < 0:
        raise ValueError(
            "transition_fallback_evidence_window must be non-negative"
        )
    if transition_fallback_min_hits < 0:
        raise ValueError(
            "transition_fallback_min_hits must be non-negative"
        )
    if (
        transition_fallback_evidence_window == 0
        and transition_fallback_min_hits != 0
    ):
        raise ValueError(
            "transition_fallback_min_hits must be 0 when "
            "transition_fallback_evidence_window is 0"
        )
    if (
        transition_fallback_evidence_window > 0
        and not 1
        <= transition_fallback_min_hits
        <= transition_fallback_evidence_window
    ):
        raise ValueError(
            "transition_fallback_min_hits must be within "
            "the evidence window"
        )
    if dino_features.ndim != 3:
        raise ValueError("dino_features must have shape [image, patch, feature]")
    if len(timeline) != dino_features.shape[0]:
        raise ValueError("timeline and feature image counts must match")
    if transition_fallback_window < 0:
        raise ValueError("transition_fallback_window must be non-negative")
    if retrain_policy not in {"on_change", "fit_once", "committee_gate"}:
        raise ValueError(f"unsupported retrain_policy: {retrain_policy}")
    if training_source not in {"cluster_history", "active_committee"}:
        raise ValueError(f"unsupported training_source: {training_source}")
    if fusion_mode not in {"dino_only", "strict_history_dino"}:
        if clip_features is None or clip_features.shape[0] != dino_features.shape[0]:
            raise ValueError("DINO and CLIP image counts must match")

    if any(
        record.get("controlled_contamination_replay") is not None
        for record in timeline
    ):
        dino_features = dino_features.clone()
        for step, record in enumerate(timeline):
            replay = record.get("controlled_contamination_replay")
            if replay is None:
                continue
            source_image_id = int(replay["source_image_id"])
            clean_image_id = int(replay["clean_image_id"])
            if source_image_id >= step or clean_image_id >= step:
                raise ValueError(
                    "controlled replay references must be historical"
                )
            patch_ids = torch.as_tensor(
                replay["patch_ids"],
                device=dino_features.device,
                dtype=torch.long,
            )
            replay_feature = dino_features[clean_image_id].clone()
            replay_feature.index_copy_(
                0,
                patch_ids,
                dino_features[source_image_id].index_select(0, patch_ids),
            )
            dino_features[step] = replay_feature

    scores = []
    audits = []
    cached_signature = None
    cached_training = None
    detect_fuser = None
    fitted_signature = None
    last_fit_step = None
    previous_active_signature = None
    active_stable_count = 0
    history_start = 0
    transition_fallback_pool_start = None
    transition_fallback_latched = False

    for step, record in enumerate(timeline):
        if record.get("state_reset_before", False):
            history_start = step
            transition_fallback_pool_start = None
            transition_fallback_latched = False
            cached_signature = None
            cached_training = None
            detect_fuser = None
            fitted_signature = None
            last_fit_step = None
            previous_active_signature = None
            active_stable_count = 0
        current_dino = dino_features[step]
        active = record["active_experts_before"]
        expert_image_ids = [expert["image_id"] for expert in active]
        active_dino_reference_features = []
        for expert in active:
            reference_feature = dino_features[int(expert["image_id"])]
            if expert.get("controlled_contamination_mode") in {
                "patch",
                "mixed",
            }:
                clean_feature = dino_features[
                    int(expert["controlled_clean_image_id"])
                ]
                patch_ids = torch.as_tensor(
                    expert.get("controlled_contaminated_patch_ids", []),
                    device=reference_feature.device,
                    dtype=torch.long,
                )
                hybrid_feature = clean_feature.clone()
                if patch_ids.numel() > 0:
                    hybrid_feature.index_copy_(
                        0,
                        patch_ids,
                        reference_feature.index_select(0, patch_ids),
                    )
                reference_feature = hybrid_feature
            active_dino_reference_features.append(reference_feature)
        active_dino_reference_tensor = (
            torch.stack(active_dino_reference_features)
            if active_dino_reference_features
            else None
        )
        active_expert_scoring_weights = [
            float(expert.get("scoring_weight", 1.0))
            for expert in active
        ]
        active_expert_global_reliabilities = [
            float(
                expert.get(
                    "global_member_reliability",
                    expert.get("member_reliability", 1.0),
                )
            )
            for expert in active
        ]
        active_expert_patch_reliabilities = [
            expert.get("scoring_patch_reliabilities")
            for expert in active
        ]
        reference_reliability_penalty = max(
            [
                float(expert.get("reference_reliability_penalty", 1.0))
                for expert in active
            ],
            default=1.0,
        )
        active_patch_reliability_tensor = None
        if any(
            values is not None
            for values in active_expert_patch_reliabilities
        ):
            active_patch_reliability_tensor = torch.stack(
                [
                    torch.as_tensor(
                        values,
                        device=current_dino.device,
                        dtype=current_dino.dtype,
                    )
                    if values is not None
                    else torch.ones(
                        current_dino.shape[0],
                        device=current_dino.device,
                        dtype=current_dino.dtype,
                    )
                    for values in active_expert_patch_reliabilities
                ]
            )
        active_committee_confidence = (
            max(active_expert_scoring_weights)
            if active_expert_scoring_weights
            else 0.0
        )
        active_signature = tuple(expert_image_ids)
        active_stable_count = (
            active_stable_count + 1
            if active_signature == previous_active_signature
            else 1
        )
        previous_active_signature = active_signature
        chosen_training_ids = [
            member["image_id"]
            for member in record["fuser_training_members_before"]
        ]
        fallback_reason = None
        effective_mode = fusion_mode
        score_audit = None
        hierarchical_history_score = None
        hierarchical_history_anchor_expert_ids = []
        reference_contributions = []
        fuser_retrained = False
        training_recomputed = False
        change_ratio = None
        provisional = record.get("provisional_expert_before")
        provisional_score_audit = None
        fallback_history_image_ids = []
        fallback_has_pre_pool_expert = False
        fallback_recent_evidence_hits = 0
        fallback_recent_evidence_steps = []
        transition_fallback_triggered = False

        if fusion_mode == "strict_history_dino":
            patch_score, fallback_reason = _strict_history_dino_fallback(
                current_dino,
                dino_features[history_start:step],
            )
            effective_mode = "strict_history_dino"
        elif not expert_image_ids:
            if fusion_mode == "clip_only":
                patch_score, fallback_reason = _strict_history_clip_fallback(
                    clip_features[step],
                    clip_features[history_start:step],
                )
                effective_mode = "clip_fallback"
            else:
                patch_score, fallback_reason = _strict_history_dino_fallback(
                    current_dino,
                    dino_features[history_start:step],
                )
                effective_mode = "dino_fallback"
        elif fusion_mode == "dino_only" and hierarchical_reliability_enabled:
            hierarchical_history_score, _ = _strict_history_dino_fallback(
                current_dino,
                dino_features[history_start:step],
            )
            if len(active) > 1:
                maximum_global_reliability = max(
                    active_expert_global_reliabilities
                )
                anchor_indices = [
                    index
                    for index, reliability in enumerate(
                        active_expert_global_reliabilities
                    )
                    if math.isclose(
                        reliability,
                        maximum_global_reliability,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ]
                anchor_index_tensor = torch.as_tensor(
                    anchor_indices,
                    device=active_dino_reference_tensor.device,
                    dtype=torch.long,
                )
                hierarchical_history_anchor_expert_ids = [
                    int(active[index]["expert_id"])
                    for index in anchor_indices
                ]
                hierarchical_history_score, _ = MSM2_online(
                    current_dino,
                    None,
                    active_dino_reference_tensor.index_select(
                        0,
                        anchor_index_tensor,
                    ),
                    None,
                    fusion_mode="dino_only",
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                    dino_patch_reliabilities=(
                        active_patch_reliability_tensor.index_select(
                            0,
                            anchor_index_tensor,
                        )
                        if active_patch_reliability_tensor is not None
                        else None
                    ),
                    dino_reference_member_ids=[
                        int(expert_image_ids[index])
                        for index in anchor_indices
                    ],
                    reliability_penalty=reference_reliability_penalty,
                )
            patch_score, score_audit, reference_contributions = (
                _hierarchical_dino_score(
                    current_dino=current_dino,
                    expert_dino=active_dino_reference_tensor,
                    expert_patch_reliabilities=(
                        active_patch_reliability_tensor
                    ),
                    expert_global_reliabilities=(
                        active_expert_global_reliabilities
                    ),
                    expert_ids=[
                        int(expert["expert_id"]) for expert in active
                    ],
                    expert_image_ids=[
                        int(image_id) for image_id in expert_image_ids
                    ],
                    history_score=hierarchical_history_score,
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                    reliability_penalty=reference_reliability_penalty,
                )
            )
            effective_mode = "dino_hierarchical_reliability"
        elif fusion_mode == "dino_only":
            full_expert_indices = [
                index
                for index, weight in enumerate(active_expert_scoring_weights)
                if weight >= 1.0
            ]
            full_expert_image_ids = [
                expert_image_ids[index] for index in full_expert_indices
            ]
            confidence_experts = [
                (index, expert_image_ids[index], weight)
                for index, weight in enumerate(active_expert_scoring_weights)
                if weight < 1.0
            ]
            (
                patch_score,
                score_audit,
                confidence_reference_audits,
                fallback_reason,
            ) = score_dino_reference_committee(
                current_dino,
                active_dino_reference_tensor,
                active_expert_scoring_weights,
                fallback_references=dino_features[history_start:step],
                patch_reliabilities=active_patch_reliability_tensor,
                reference_member_ids=expert_image_ids,
                topmin_min=topmin_min,
                topmin_max=topmin_max,
                reliability_penalty=reference_reliability_penalty,
            )
            remaining_blend_weight = 1.0
            confidence_contributions = []
            for (
                confidence_index,
                confidence_weight,
                confidence_score_audit,
            ) in reversed(confidence_reference_audits):
                contribution_weight = (
                    float(confidence_weight) * remaining_blend_weight
                )
                confidence_contributions.extend(
                    _reference_contributions(
                        confidence_score_audit,
                        [active[confidence_index]["expert_id"]],
                        blend_weight=contribution_weight,
                    )
                )
                remaining_blend_weight *= 1.0 - float(confidence_weight)
            if score_audit is not None:
                reference_contributions.extend(
                    _reference_contributions(
                        score_audit,
                        [
                            active[index]["expert_id"]
                            for index in full_expert_indices
                        ],
                        blend_weight=remaining_blend_weight,
                    )
                )
            reference_contributions.extend(
                reversed(confidence_contributions)
            )
            if confidence_experts:
                effective_mode = "dino_confidence_blend"
        elif fusion_mode == "clip_only":
            patch_score, score_audit = MSM2_online(
                current_dino,
                clip_features[step],
                dino_features[expert_image_ids],
                clip_features[expert_image_ids],
                fusion_mode="clip_only",
                topmin_min=topmin_min,
                topmin_max=topmin_max,
            )
        else:
            if training_source == "active_committee":
                member_ids = list(expert_image_ids)
                member_steps = list(expert_image_ids)
            else:
                members = record["fuser_training_members_before"]
                member_ids = [member["image_id"] for member in members]
                member_steps = [member["step"] for member in members]
            chosen_training_ids = list(member_ids)
            signature = tuple(member_ids)
            committee_allowed = True
            gate_reason = None
            if retrain_policy == "committee_gate":
                if len(member_ids) < committee_min_experts:
                    committee_allowed = False
                    gate_reason = "insufficient_active_committee"
                elif (
                    cached_training is None
                    and active_stable_count < committee_stable_steps
                ):
                    committee_allowed = False
                    gate_reason = "committee_not_stable"

            if retrain_policy == "committee_gate":
                if cached_training is None:
                    should_recompute = committee_allowed
                else:
                    overlap = len(set(fitted_signature or ()) & set(signature))
                    denominator = max(len(fitted_signature or ()), len(signature), 1)
                    change_ratio = 1.0 - overlap / denominator
                    cooldown_ready = (
                        last_fit_step is None
                        or step - last_fit_step >= committee_retrain_cooldown
                    )
                    should_recompute = (
                        committee_allowed
                        and active_stable_count >= committee_stable_steps
                        and change_ratio >= committee_change_ratio
                        and cooldown_ready
                    )
            else:
                change_ratio = None
                should_recompute = signature != cached_signature and (
                    retrain_policy == "on_change"
                    or cached_training is None
                    or not cached_training.get("fitted", False)
                )
            if should_recompute:
                training_patch_reliabilities = None
                if training_source == "active_committee":
                    training_patch_reliabilities = (
                        active_patch_reliability_tensor
                    )
                elif members:
                    training_patch_reliabilities = torch.stack(
                        [
                            torch.as_tensor(
                                member.get(
                                    "scoring_patch_reliabilities",
                                    member.get(
                                        "patch_reliabilities",
                                        torch.ones(current_dino.shape[0]),
                                    ),
                                ),
                                device=current_dino.device,
                                dtype=current_dino.dtype,
                            )
                            for member in members
                        ]
                    )
                if fusion_mode == "fixed":
                    cached_training = build_causal_fuser_training_data(
                        dino_features[member_ids],
                        clip_features[member_ids],
                        member_ids,
                        member_steps,
                        current_step=step,
                        dino_weight=dino_weight,
                        clip_weight=clip_weight,
                        topmin_min=topmin_min,
                        topmin_max=topmin_max,
                        training_dino_patch_reliabilities=(
                            training_patch_reliabilities
                        ),
                    )
                    cached_training["fitted"] = False
                else:
                    detect_fuser = linear_model.SGDOneClassSVM(
                        random_state=42,
                        nu=0.5,
                        max_iter=1000,
                    )
                    cached_training = fit_causal_detect_fuser(
                        detect_fuser,
                        dino_features[member_ids],
                        clip_features[member_ids],
                        member_ids,
                        member_steps,
                        current_step=step,
                        dino_weight=dino_weight,
                        clip_weight=clip_weight,
                        topmin_min=topmin_min,
                        topmin_max=topmin_max,
                        training_dino_patch_reliabilities=(
                            training_patch_reliabilities
                        ),
                    )
                cached_signature = signature
                training_recomputed = True
                fuser_retrained = fusion_mode == "fuser"
                if cached_training.get("fitted", False):
                    fitted_signature = signature
                    last_fit_step = step

            dino_scale = (
                cached_training.get("dino_scale") if cached_training else None
            )
            clip_scale = (
                cached_training.get("clip_scale") if cached_training else None
            )
            if (
                committee_allowed
                and fusion_mode == "fuser"
                and cached_training
                and cached_training["fitted"]
            ):
                patch_score, score_audit = MSM2_online(
                    current_dino,
                    clip_features[step],
                    dino_features[expert_image_ids],
                    clip_features[expert_image_ids],
                    detect_fuser=detect_fuser,
                    fusion_mode="fuser",
                    dino_weight=dino_weight,
                    clip_weight=clip_weight,
                    epsilon=epsilon,
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                    dino_patch_reliabilities=(
                        active_patch_reliability_tensor
                    ),
                    dino_reference_member_ids=expert_image_ids,
                    reliability_penalty=reference_reliability_penalty,
                )
            elif (
                committee_allowed
                and dino_scale is not None
                and clip_scale is not None
            ):
                patch_score, score_audit = MSM2_online(
                    current_dino,
                    clip_features[step],
                    dino_features[expert_image_ids],
                    clip_features[expert_image_ids],
                    fusion_mode="fixed",
                    dino_scale=dino_scale,
                    clip_scale=clip_scale,
                    dino_weight=dino_weight,
                    clip_weight=clip_weight,
                    epsilon=epsilon,
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                    dino_patch_reliabilities=(
                        active_patch_reliability_tensor
                    ),
                    dino_reference_member_ids=expert_image_ids,
                    reliability_penalty=reference_reliability_penalty,
                )
                effective_mode = "fixed" if fusion_mode == "fixed" else "fixed_fallback"
                fallback_reason = cached_training["reason"]
            else:
                if gate_reason is not None:
                    patch_score, score_audit = MSM2_online(
                        current_dino,
                        None,
                        dino_features[expert_image_ids],
                        None,
                        fusion_mode="dino_only",
                        topmin_min=topmin_min,
                        topmin_max=topmin_max,
                        dino_patch_reliabilities=(
                            active_patch_reliability_tensor
                        ),
                        dino_reference_member_ids=expert_image_ids,
                        reliability_penalty=reference_reliability_penalty,
                    )
                    fallback_reason = gate_reason
                    effective_mode = "dino_gate"
                else:
                    patch_score, fallback_reason = _strict_history_dino_fallback(
                        current_dino,
                        dino_features[history_start:step],
                    )
                    effective_mode = "dino_fallback"

        if (
            provisional is None
            or provisional.get("state") != "collecting_channel_gate"
        ):
            transition_fallback_pool_start = None
            transition_fallback_latched = False

        if fusion_mode == "dino_only" and provisional is not None:
            provisional_image_ids = [
                int(image_id)
                for image_id in provisional.get(
                    "image_ids", [provisional["image_id"]]
                )
            ]
            if any(image_id >= step for image_id in provisional_image_ids):
                raise ValueError(
                    "provisional expert must be strictly historical"
                )

            provisional_state = provisional.get("state")
            provisional_start_step = provisional.get("start_step")
            has_enough_novel_evidence = False

            if (
                provisional_state == "collecting_channel_gate"
                and provisional_start_step is not None
            ):
                provisional_start_step = int(provisional_start_step)
                if (
                    transition_fallback_pool_start
                    != provisional_start_step
                ):
                    transition_fallback_pool_start = (
                        provisional_start_step
                    )
                    transition_fallback_latched = False

                fallback_has_pre_pool_expert = any(
                    int(image_id) < provisional_start_step
                    for image_id in expert_image_ids
                )
                rolling_gate_enabled = (
                    transition_fallback_evidence_window > 0
                    and transition_fallback_min_hits > 0
                )

                if rolling_gate_enabled:
                    evidence_start = max(
                        history_start,
                        provisional_start_step,
                        step - transition_fallback_evidence_window,
                    )
                    fallback_recent_evidence_steps = [
                        history_step
                        for history_step in range(evidence_start, step)
                        if timeline[history_step].get("candidate_route")
                        == "new_distribution"
                    ]
                    fallback_recent_evidence_hits = len(
                        fallback_recent_evidence_steps
                    )
                    evidence_confirmed = (
                        fallback_recent_evidence_hits
                        >= transition_fallback_min_hits
                    )

                    if transition_fallback_latch and evidence_confirmed:
                        transition_fallback_latched = True

                    has_enough_novel_evidence = (
                        transition_fallback_latched
                        if transition_fallback_latch
                        else evidence_confirmed
                    )
                else:
                    has_enough_novel_evidence = True

            should_use_transition_fallback = (
                provisional_state == "collecting_channel_gate"
                and transition_fallback_window > 0
                and (
                    not transition_fallback_require_pre_pool_expert
                    or fallback_has_pre_pool_expert
                )
                and has_enough_novel_evidence
            )

            if should_use_transition_fallback:
                fallback_start = max(
                    history_start,
                    step - transition_fallback_window,
                )
                fallback_history_image_ids = list(
                    range(fallback_start, step)
                )
                patch_score, fallback_reason = (
                    _strict_history_dino_fallback(
                        current_dino,
                        dino_features[fallback_start:step],
                    )
                )
                fallback_reason = "category_transition_recent_history"
                effective_mode = "recent_history_dino_fallback"
                transition_fallback_triggered = True
            elif provisional_state in {
                "collecting_irbank",
                "confirmed_irbank",
            }:
                memory_members = provisional.get("memory_members", [])
                member_features = [
                    dino_features[int(member["image_id"])]
                    for member in memory_members
                ]
                member_patch_ids = [
                    member.get("patch_ids", []) for member in memory_members
                ]
                memory_score, memory_valid = _positioned_memory_scores(
                    current_dino,
                    member_features,
                    member_patch_ids,
                    grid_size=math.isqrt(current_dino.shape[0]),
                    position_radius=position_radius,
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                )
                base_before_memory = patch_score
                patch_score = torch.where(
                    memory_valid,
                    memory_score,
                    patch_score,
                )
                effective_mode = "dino_channel_irbank"
                provisional_score_audit = {
                    "memory_patch_count": sum(
                        len(patch_ids) for patch_ids in member_patch_ids
                    ),
                    "memory_valid_fraction": float(
                        memory_valid.float().mean().cpu()
                    ),
                    "memory_score_mean": float(
                        memory_score[memory_valid].mean().cpu()
                    ),
                    "memory_score_max": float(
                        memory_score[memory_valid].max().cpu()
                    ),
                    "base_score_mean_before_memory": float(
                        base_before_memory.mean().cpu()
                    ),
                    "base_score_max_before_memory": float(
                        base_before_memory.max().cpu()
                    ),
                }
            else:
                provisional_weight = float(provisional["weight"])
                if not 0.0 <= provisional_weight <= 1.0:
                    raise ValueError(
                        "provisional expert weight must be within [0, 1]"
                    )
                provisional_score, provisional_score_audit = MSM2_online(
                    current_dino,
                    None,
                    dino_features[provisional_image_ids],
                    None,
                    fusion_mode="dino_only",
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                )
                patch_score = (
                    (1.0 - provisional_weight) * patch_score
                    + provisional_weight * provisional_score
                )
                effective_mode = "dino_provisional"

        if fusion_mode != "dino_only" and score_audit is not None:
            reference_contributions = _reference_contributions(
                score_audit,
                [expert["expert_id"] for expert in active],
            )

        if not torch.isfinite(patch_score).all():
            reference_contributions = []
            if fusion_mode == "clip_only":
                patch_score, fallback_reason = _strict_history_clip_fallback(
                    clip_features[step],
                    clip_features[history_start:step],
                )
                effective_mode = "clip_fallback"
            else:
                patch_score, fallback_reason = _strict_history_dino_fallback(
                    current_dino,
                    dino_features[history_start:step],
                )
                effective_mode = "dino_fallback"
            fallback_reason = f"non_finite_score:{fallback_reason}"

        controlled_contaminated_image_ids = {
            int(event["source_image_id"])
            for previous_record in timeline[:step]
            for event in [
                previous_record.get("controlled_contamination") or {}
            ]
            if event.get("injected")
            and event.get("source_image_id") is not None
        }
        def image_score(score):
            if image_score_size is None:
                return float(score.max().detach().cpu())
            patch_count = int(score.numel())
            side = math.isqrt(patch_count)
            if side * side != patch_count:
                raise ValueError(
                    "image_score_size requires a square patch map"
                )
            resized = torch.nn.functional.interpolate(
                score.reshape(1, 1, side, side),
                size=image_score_size,
                mode="bilinear",
                align_corners=True,
            )
            return float(resized.max().detach().cpu())

        score_all = image_score(patch_score)
        reentry_expert_image_ids = {
            int(expert["image_id"])
            for expert in active
            if "history_reentry" in str(expert.get("admission_route", ""))
        }
        reentry_counterfactual_score = None
        reentry_score_regret = None
        if (
            fusion_mode == "dino_only"
            and provisional is None
            and reentry_expert_image_ids
        ):
            reentry_counterfactual_patch_score = None
            if hierarchical_reliability_enabled:
                reentry_clean_indices = [
                    index
                    for index, image_id in enumerate(expert_image_ids)
                    if int(image_id) not in reentry_expert_image_ids
                ]
                if (
                    reentry_clean_indices
                    and hierarchical_history_score is not None
                ):
                    reentry_clean_index_tensor = torch.as_tensor(
                        reentry_clean_indices,
                        device=active_dino_reference_tensor.device,
                        dtype=torch.long,
                    )
                    reentry_counterfactual_patch_score, _, _ = (
                        _hierarchical_dino_score(
                            current_dino=current_dino,
                            expert_dino=(
                                active_dino_reference_tensor.index_select(
                                    0, reentry_clean_index_tensor
                                )
                            ),
                            expert_patch_reliabilities=(
                                active_patch_reliability_tensor.index_select(
                                    0, reentry_clean_index_tensor
                                )
                                if active_patch_reliability_tensor is not None
                                else None
                            ),
                            expert_global_reliabilities=[
                                active_expert_global_reliabilities[index]
                                for index in reentry_clean_indices
                            ],
                            expert_ids=[
                                int(active[index]["expert_id"])
                                for index in reentry_clean_indices
                            ],
                            expert_image_ids=[
                                int(expert_image_ids[index])
                                for index in reentry_clean_indices
                            ],
                            history_score=hierarchical_history_score,
                            topmin_min=topmin_min,
                            topmin_max=topmin_max,
                            reliability_penalty=(
                                reference_reliability_penalty
                            ),
                        )
                    )
            else:
                reentry_counterfactual_patch_score = (
                    _counterfactual_dino_score_without_references(
                    current_dino=current_dino,
                    dino_features=dino_features,
                    active=active,
                    excluded_image_ids=reentry_expert_image_ids,
                    history_start=history_start,
                    step=step,
                    topmin_min=topmin_min,
                    topmin_max=topmin_max,
                    reliability_penalty=reference_reliability_penalty,
                    )
                )
            if reentry_counterfactual_patch_score is not None:
                reentry_counterfactual_score = image_score(
                    reentry_counterfactual_patch_score
                )
                reentry_score_regret = (
                    score_all - reentry_counterfactual_score
                )
        score_without_contaminated_reference = None
        pollution_score_shift = None
        if (
            fusion_mode == "dino_only"
            and provisional is None
            and controlled_contaminated_image_ids
            and any(
                int(image_id) in controlled_contaminated_image_ids
                for image_id in expert_image_ids
            )
        ):
            if hierarchical_reliability_enabled:
                clean_indices = [
                    index
                    for index, image_id in enumerate(expert_image_ids)
                    if int(image_id)
                    not in controlled_contaminated_image_ids
                ]
                counterfactual_patch_score = None
                if (
                    clean_indices
                    and hierarchical_history_score is not None
                ):
                    clean_index_tensor = torch.as_tensor(
                        clean_indices,
                        device=active_dino_reference_tensor.device,
                        dtype=torch.long,
                    )
                    counterfactual_patch_score, _, _ = (
                        _hierarchical_dino_score(
                            current_dino=current_dino,
                            expert_dino=(
                                active_dino_reference_tensor.index_select(
                                    0,
                                    clean_index_tensor,
                                )
                            ),
                            expert_patch_reliabilities=(
                                active_patch_reliability_tensor.index_select(
                                    0,
                                    clean_index_tensor,
                                )
                                if active_patch_reliability_tensor is not None
                                else None
                            ),
                            expert_global_reliabilities=[
                                active_expert_global_reliabilities[index]
                                for index in clean_indices
                            ],
                            expert_ids=[
                                int(active[index]["expert_id"])
                                for index in clean_indices
                            ],
                            expert_image_ids=[
                                int(expert_image_ids[index])
                                for index in clean_indices
                            ],
                            history_score=hierarchical_history_score,
                            topmin_min=topmin_min,
                            topmin_max=topmin_max,
                            reliability_penalty=(
                                reference_reliability_penalty
                            ),
                        )
                    )
            else:
                counterfactual_patch_score = (
                    _counterfactual_dino_score_without_references(
                        current_dino=current_dino,
                        dino_features=dino_features,
                        active=active,
                        excluded_image_ids=(
                            controlled_contaminated_image_ids
                        ),
                        history_start=history_start,
                        step=step,
                        topmin_min=topmin_min,
                        topmin_max=topmin_max,
                        reliability_penalty=reference_reliability_penalty,
                    )
                )
            if counterfactual_patch_score is not None:
                score_without_contaminated_reference = image_score(
                    counterfactual_patch_score
                )
                pollution_score_shift = (
                    score_all - score_without_contaminated_reference
                )

        audit = {
            "step": step,
            "segment_start": history_start,
            "state_reset_before": record.get("state_reset_before", False),
            "requested_mode": fusion_mode,
            "effective_mode": effective_mode,
            "active_expert_ids": [expert["expert_id"] for expert in active],
            "expert_image_ids": expert_image_ids,
            "active_expert_scoring_weights": active_expert_scoring_weights,
            "active_expert_member_reliabilities": [
                float(expert.get("member_reliability", 1.0))
                for expert in active
            ],
            "active_expert_global_reliabilities": (
                active_expert_global_reliabilities
            ),
            "hierarchical_diagnostics": (
                score_audit.get("hierarchical_diagnostics")
                if score_audit is not None
                else None
            ),
            "hierarchical_history_anchor_expert_ids": (
                hierarchical_history_anchor_expert_ids
            ),
            "active_expert_patch_reliability_means": [
                (
                    float(np.mean(values))
                    if values is not None
                    else 1.0
                )
                for values in active_expert_patch_reliabilities
            ],
            "scored_reference_expert_ids": [
                int(item["expert_id"])
                for item in reference_contributions
            ],
            "selected_reference_contributions": reference_contributions,
            "controlled_contaminated_image_ids": sorted(
                controlled_contaminated_image_ids
            ),
            "score_all": score_all,
            "reentry_expert_image_ids": sorted(reentry_expert_image_ids),
            "reentry_counterfactual_score": reentry_counterfactual_score,
            "reentry_score_regret": reentry_score_regret,
            "score_without_contaminated_reference": (
                score_without_contaminated_reference
            ),
            "pollution_score_shift": pollution_score_shift,
            "image_score_size": image_score_size,
            "controlled_contamination": record.get(
                "controlled_contamination"
            ),
            "active_committee_confidence": active_committee_confidence,
            "provisional_expert_image_id": (
                int(provisional["image_id"])
                if provisional is not None
                else None
            ),
            "provisional_expert_image_ids": (
                [
                    int(image_id)
                    for image_id in provisional.get(
                        "image_ids", [provisional["image_id"]]
                    )
                ]
                if provisional is not None
                else []
            ),
            "provisional_expert_weight": (
                float(provisional["weight"])
                if provisional is not None
                else None
            ),
            "provisional_pool_size": (
                int(provisional["pool_size"])
                if provisional is not None
                else 0
            ),
            "provisional_state": (
                provisional.get("state")
                if provisional is not None
                else None
            ),
            "fuser_training_image_ids": (
                chosen_training_ids
            ),
            "fuser_retrained": fuser_retrained,
            "training_recomputed": training_recomputed,
            "retrain_policy": retrain_policy,
            "training_source": training_source,
            "active_committee_stable_count": active_stable_count,
            "committee_change_ratio": (
                change_ratio
            ),
            "fallback_reason": fallback_reason,
            "fallback_history_image_ids": fallback_history_image_ids,
            "fallback_has_pre_pool_expert": fallback_has_pre_pool_expert,
            "fallback_recent_evidence_hits": fallback_recent_evidence_hits,
            "fallback_recent_evidence_steps": fallback_recent_evidence_steps,
            "transition_fallback_evidence_window": (
                transition_fallback_evidence_window
            ),
            "transition_fallback_min_hits": transition_fallback_min_hits,
            "transition_fallback_latched": transition_fallback_latched,
            "transition_fallback_triggered": transition_fallback_triggered,
            "training_image_count": (
                len(cached_training["image_ids"])
                if cached_training is not None
                else 0
            ),
            "training_patch_pair_count": (
                int(cached_training["train_pairs"].shape[0])
                if cached_training is not None
                and cached_training.get("train_pairs") is not None
                else 0
            ),
        }
        if score_audit is not None:
            audit.update(_distance_audit(score_audit))
        if provisional_score_audit is not None and "memory_patch_count" in provisional_score_audit:
            audit.update(provisional_score_audit)
        elif provisional_score_audit is not None:
            provisional_distance_audit = _distance_audit(
                provisional_score_audit
            )
            audit.update(
                {
                    f"provisional_{key}": value
                    for key, value in provisional_distance_audit.items()
                }
            )
        if score_audit is not None and score_audit.get("fuser_score") is not None:
            audit.update(
                {
                    "fuser_min": score_audit["fuser_min"],
                    "fuser_max": score_audit["fuser_max"],
                    "fuser_mean": score_audit["fuser_mean"],
                    "fuser_positive_ratio": score_audit["fuser_positive_ratio"],
                    "fuser_negative_ratio": score_audit["fuser_negative_ratio"],
                }
            )
        scores.append(patch_score)
        audits.append(audit)
        if "score_audit" in locals():
            del score_audit

    return torch.stack(scores), audits
