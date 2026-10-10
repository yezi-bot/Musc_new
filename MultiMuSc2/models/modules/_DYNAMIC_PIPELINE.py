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
)


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


def build_dynamic_committee_timeline(
    committee_features,
    device,
    position_radius=1,
    channel_distance_quantile=0.7,
    reliability_alpha=0.5,
    channel_ttl=5,
    mature_span=3.0,
    density_k=5,
    ms_short_windows=(8, 16, 32),
    manager_kwargs=None,
    reset_steps=None,
):
    if committee_features.ndim != 3:
        raise ValueError("committee_features must have shape [image, patch, feature]")
    ms_short_windows = tuple(int(window) for window in ms_short_windows)
    grid_size = math.isqrt(committee_features.shape[1])
    if grid_size * grid_size != committee_features.shape[1]:
        raise ValueError("committee patch count must form a square grid")

    reset_steps = set(reset_steps or [])
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
            segment_start = step
        current = committee_features[step]
        active_before = manager.active_experts_before_step()
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

        expert_supports = {}
        for expert in active_before:
            expert_features = committee_features[expert["image_id"]]
            support, _ = _channel_support(
                memory,
                expert_features,
                grid_size,
                distance_threshold,
            )
            expert_supports[expert["expert_id"]] = support

        event = manager.advance(
            step=step,
            image_id=step,
            dino_patch_features=current,
            ms_score=ms_score,
            channel_support=channel_support,
            expert_channel_supports=expert_supports,
            provisional_channel_support=association_profile[
                "recent_provisional_match_fraction"
            ],
            ms_short_score=ms_short,
            ms_short_ratio=ms_short_ratio,
        )

        if current_distances is not None:
            finite = current_distances[torch.isfinite(current_distances)]
            distance_history.extend(finite.tolist())

        _, reliability = memory.patch_reliability(current, grid_size)
        soft_reliability = reliability_alpha + (1.0 - reliability_alpha) * reliability
        memory.update(
            current,
            image_id=step,
            grid_size=grid_size,
            patch_reliabilities=soft_reliability,
        )
        historical_features.append(current.detach().cpu())

        timeline.append(
            {
                "step": step,
                "segment_start": segment_start,
                "state_reset_before": state_reset_before,
                "active_experts_before": active_before,
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
                "candidate_route": event["candidate_route"],
                "admission_route": event["admission_route"],
                "novel_quarantine": event["novel_quarantine"],
                "novel_support_threshold": event[
                    "novel_support_threshold"
                ],
                "novel_ms_threshold": event["novel_ms_threshold"],
                "expert_channel_supports": expert_supports,
                "admitted_expert_id": event["admitted_expert_id"],
                "deleted_expert_ids": event["deleted_expert_ids"],
                "active_experts_after": manager.active_experts_before_step(),
                "active_expert_ids_after": event["active_expert_ids_after_step"],
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
    return result


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
):
    if dino_features.ndim != 3:
        raise ValueError("dino_features must have shape [image, patch, feature]")
    if len(timeline) != dino_features.shape[0]:
        raise ValueError("timeline and feature image counts must match")
    if retrain_policy not in {"on_change", "fit_once", "committee_gate"}:
        raise ValueError(f"unsupported retrain_policy: {retrain_policy}")
    if training_source not in {"cluster_history", "active_committee"}:
        raise ValueError(f"unsupported training_source: {training_source}")
    if fusion_mode not in {"dino_only", "strict_history_dino"}:
        if clip_features is None or clip_features.shape[0] != dino_features.shape[0]:
            raise ValueError("DINO and CLIP image counts must match")

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

    for step, record in enumerate(timeline):
        if record.get("state_reset_before", False):
            history_start = step
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
        fuser_retrained = False
        training_recomputed = False
        change_ratio = None
        provisional = record.get("provisional_expert_before")
        provisional_score_audit = None

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
        elif fusion_mode == "dino_only":
            patch_score, score_audit = MSM2_online(
                current_dino,
                None,
                dino_features[expert_image_ids],
                None,
                fusion_mode="dino_only",
                topmin_min=topmin_min,
                topmin_max=topmin_max,
            )
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
                    )
                    fallback_reason = gate_reason
                    effective_mode = "dino_gate"
                else:
                    patch_score, fallback_reason = _strict_history_dino_fallback(
                        current_dino,
                        dino_features[history_start:step],
                    )
                    effective_mode = "dino_fallback"

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

        if not torch.isfinite(patch_score).all():
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

        audit = {
            "step": step,
            "segment_start": history_start,
            "state_reset_before": record.get("state_reset_before", False),
            "requested_mode": fusion_mode,
            "effective_mode": effective_mode,
            "active_expert_ids": [expert["expert_id"] for expert in active],
            "expert_image_ids": expert_image_ids,
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
        if provisional_score_audit is not None:
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
