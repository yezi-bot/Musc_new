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


def _strict_history_dino_fallback(current_dino, dino_history):
    if dino_history.shape[0] == 0:
        return torch.zeros(
            current_dino.shape[0],
            device=current_dino.device,
            dtype=torch.float32,
        ), "first_image_unavailable"
    return aggregate_reference_distances(
        current_dino,
        dino_history,
        topmin_min=0,
        topmin_max=0.3,
    ), "strict_history_dino"


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
):
    if dino_features.ndim != 3:
        raise ValueError("dino_features must have shape [image, patch, feature]")
    if len(timeline) != dino_features.shape[0]:
        raise ValueError("timeline and feature image counts must match")
    if fusion_mode != "dino_only":
        if clip_features is None or clip_features.shape[0] != dino_features.shape[0]:
            raise ValueError("DINO and CLIP image counts must match")

    scores = []
    audits = []
    cached_signature = None
    cached_training = None
    detect_fuser = None

    for step, record in enumerate(timeline):
        current_dino = dino_features[step]
        active = record["active_experts_before"]
        expert_image_ids = [expert["image_id"] for expert in active]
        fallback_reason = None
        effective_mode = fusion_mode
        fuser_retrained = False

        if not expert_image_ids:
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
        else:
            members = record["fuser_training_members_before"]
            member_ids = [member["image_id"] for member in members]
            member_steps = [member["step"] for member in members]
            signature = tuple(member_ids)
            if signature != cached_signature:
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
                fuser_retrained = True

            dino_scale = cached_training.get("dino_scale")
            clip_scale = cached_training.get("clip_scale")
            if fusion_mode == "fuser" and cached_training["fitted"]:
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
            elif dino_scale is not None and clip_scale is not None:
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
                patch_score, fallback_reason = _strict_history_dino_fallback(
                    current_dino,
                    dino_features[:step],
                )
                effective_mode = "dino_fallback"

        if not torch.isfinite(patch_score).all():
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
                [member["image_id"] for member in record["fuser_training_members_before"]]
            ),
            "fuser_retrained": fuser_retrained,
            "fallback_reason": fallback_reason,
        }
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
