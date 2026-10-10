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


def _image_msm_score(current, historical_features, reference_chunk_size=8):
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
    patch_scores = interval_average(
        torch.cat(patch_to_image, dim=1),
        topmin_min=0,
        topmin_max=0.3,
    )
    return float(patch_scores.max().cpu())


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
    manager_kwargs=None,
):
    if committee_features.ndim != 3:
        raise ValueError("committee_features must have shape [image, patch, feature]")
    grid_size = math.isqrt(committee_features.shape[1])
    if grid_size * grid_size != committee_features.shape[1]:
        raise ValueError("committee patch count must form a square grid")

    memory = ChannelMemory(
        max_ttl=channel_ttl,
        mature_span=mature_span,
        density_k=density_k,
        device=device,
        position_radius=position_radius,
    )
    manager = DynamicExpertManager(**(manager_kwargs or {}))
    historical_features = []
    distance_history = []
    timeline = []

    for step in range(committee_features.shape[0]):
        current = committee_features[step]
        active_before = manager.active_experts_before_step()
        training_members = manager.fuser_training_members(step)
        ms_score = _image_msm_score(current, historical_features)
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
                "active_experts_before": active_before,
                "fuser_training_members_before": training_members,
                "ms_score": ms_score,
                "channel_distance_threshold": distance_threshold,
                "channel_support": channel_support,
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
    committee_min_experts=4,
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
    if fusion_mode != "dino_only":
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

    for step, record in enumerate(timeline):
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
        fuser_retrained = False
        training_recomputed = False
        change_ratio = None

        if not expert_image_ids:
            if fusion_mode == "clip_only":
                patch_score, fallback_reason = _strict_history_clip_fallback(
                    clip_features[step],
                    clip_features[:step],
                )
                effective_mode = "clip_fallback"
            else:
                patch_score, fallback_reason = _strict_history_dino_fallback(
                    current_dino,
                    dino_features[:step],
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
                        dino_features[:step],
                    )
                    effective_mode = "dino_fallback"

        if not torch.isfinite(patch_score).all():
            if fusion_mode == "clip_only":
                patch_score, fallback_reason = _strict_history_clip_fallback(
                    clip_features[step],
                    clip_features[:step],
                )
                effective_mode = "clip_fallback"
            else:
                patch_score, fallback_reason = _strict_history_dino_fallback(
                    current_dino,
                    dino_features[:step],
                )
                effective_mode = "dino_fallback"
            fallback_reason = f"non_finite_score:{fallback_reason}"

        audit = {
            "step": step,
            "requested_mode": fusion_mode,
            "effective_mode": effective_mode,
            "active_expert_ids": [expert["expert_id"] for expert in active],
            "expert_image_ids": expert_image_ids,
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
        if "score_audit" in locals():
            audit.update(_distance_audit(score_audit))
        if "score_audit" in locals() and score_audit.get("fuser_score") is not None:
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
