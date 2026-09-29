"""Validate stage 1 image-level Channel support for dynamic EX-MSM."""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from validate_density_bottle import ChannelMemory, extract_features, load_samples, write_csv


def compute_online_dino_msm_image_score(
    current, reference_features, device, reference_chunk_size=8
):
    """Score one image against strictly earlier images in the current stream."""
    if reference_chunk_size < 1:
        raise ValueError("reference_chunk_size must be at least 1")
    if not reference_features:
        return None

    current = current.float().to(device)
    patch_count = current.shape[0]
    patch_to_image = []
    for start in range(0, len(reference_features), reference_chunk_size):
        chunk = torch.stack(
            reference_features[start : start + reference_chunk_size]
        ).float().to(device)
        distances = torch.cdist(current, chunk.reshape(-1, chunk.shape[-1]))
        distances = distances.reshape(patch_count, chunk.shape[0], patch_count).amin(dim=2)
        patch_to_image.append(distances)
    patch_to_image = torch.cat(patch_to_image, dim=1)
    k_max = max(1, int(patch_to_image.shape[1] * 0.3))
    patch_scores = torch.topk(
        patch_to_image, k=k_max, dim=1, largest=False, sorted=False
    ).values.mean(dim=1)
    return float(patch_scores.max().cpu())


def compute_patch_channel_distances(
    current, mature_channels, grid_size, device, position_radius=None
):
    """Return each patch's nearest mature seed distance under an optional position window."""
    seeds = torch.stack([channel.seed for channel in mature_channels]).float().to(device)
    distances = torch.cdist(current.float().to(device), seeds)
    if position_radius is not None:
        patch_ids = torch.arange(current.shape[0], device=device)
        channel_ids = torch.tensor(
            [channel.patch_ids[-1] for channel in mature_channels], device=device
        )
        patch_rows = patch_ids[:, None] // grid_size
        patch_cols = patch_ids[:, None] % grid_size
        channel_rows = channel_ids[None, :] // grid_size
        channel_cols = channel_ids[None, :] % grid_size
        valid = (patch_rows - channel_rows).abs() <= position_radius
        valid &= (patch_cols - channel_cols).abs() <= position_radius
        distances = distances.masked_fill(~valid, float("inf"))
    return distances.amin(dim=1).cpu()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def read_stage1_records(path):
    records = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                {
                    "stream_seed": int(row["stream_seed"]),
                    "stream_step": int(row["stream_step"]),
                    "image_id": int(row["image_id"]),
                    "image_type": row["image_type"],
                    "method": row["method"],
                    "reliable_mature_channels": int(row["reliable_mature_channels"]),
                    "online_threshold": (
                        float(row["online_threshold"]) if row["online_threshold"] else None
                    ),
                    "channel_support": (
                        float(row["channel_support"]) if row["channel_support"] else None
                    ),
                    "ms_score": float(row["ms_score"]) if row["ms_score"] else None,
                    "ms_score_source": row["ms_score_source"],
                }
            )
    return records


def read_stage2_candidates(path):
    candidates = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidates.append(
                {
                    "stream_seed": int(row["stream_seed"]),
                    "stream_step": int(row["stream_step"]),
                    "image_id": int(row["image_id"]),
                    "image_type": row["image_type"],
                    "ms_score": float(row["ms_score"]),
                    "channel_support": float(row["channel_support"]),
                    "ms_threshold": float(row["ms_threshold"]),
                    "support_threshold": float(row["support_threshold"]),
                    "is_candidate": row["is_candidate"].lower() == "true",
                    "expert_admitted": row["expert_admitted"].lower() == "true",
                    "candidate_index": int(row["candidate_index"]),
                }
            )
    return candidates


def read_stage3_experts(path):
    experts = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            experts.append(
                {
                    "stream_seed": int(row["stream_seed"]),
                    "image_id": int(row["image_id"]),
                    "image_type": row["image_type"],
                    "admission_trigger_step": int(row["admission_trigger_step"]),
                    "expert_index": int(row["expert_index"]),
                    "cluster_id": int(row["cluster_id"]),
                    "cluster_support_at_admission": int(
                        row["cluster_support_at_admission"]
                    ),
                    "cluster_max_support_gap": int(row["cluster_max_support_gap"]),
                }
            )
    return experts


def run_stage2_candidate_buffer(
    stage1_records,
    output_dir,
    ms_quantile=0.3,
    support_quantile=0.7,
    support_method="local",
):
    """Gate images into a buffer without admitting any image as an expert."""
    if not 0.0 <= ms_quantile <= 1.0:
        raise ValueError("ms_quantile must be within [0, 1]")
    if not 0.0 <= support_quantile <= 1.0:
        raise ValueError("support_quantile must be within [0, 1]")

    selected = [row for row in stage1_records if row["method"] == support_method]
    stream_seeds = sorted({row["stream_seed"] for row in selected})
    decisions = []
    candidate_buffer = []
    per_stream = {}
    for stream_seed in stream_seeds:
        stream_rows = sorted(
            (row for row in selected if row["stream_seed"] == stream_seed),
            key=lambda row: row["stream_step"],
        )
        ms_history = []
        support_history = []
        stream_candidates = 0
        for row in stream_rows:
            ms_threshold = (
                float(np.quantile(ms_history, ms_quantile)) if ms_history else None
            )
            support_threshold = (
                float(np.quantile(support_history, support_quantile))
                if support_history
                else None
            )
            ms_score = row["ms_score"]
            channel_support = row["channel_support"]
            is_candidate = bool(
                ms_score is not None
                and channel_support is not None
                and ms_threshold is not None
                and support_threshold is not None
                and ms_score <= ms_threshold
                and channel_support >= support_threshold
            )
            decision = {
                "stream_seed": stream_seed,
                "stream_step": row["stream_step"],
                "image_id": row["image_id"],
                "image_type": row["image_type"],
                "ms_score": ms_score,
                "channel_support": channel_support,
                "ms_threshold": ms_threshold,
                "support_threshold": support_threshold,
                "is_candidate": is_candidate,
                "expert_admitted": False,
            }
            decisions.append(decision)
            if is_candidate:
                candidate = dict(decision)
                candidate["candidate_index"] = stream_candidates
                candidate_buffer.append(candidate)
                stream_candidates += 1
            if ms_score is not None and np.isfinite(ms_score):
                ms_history.append(ms_score)
            if channel_support is not None and np.isfinite(channel_support):
                support_history.append(channel_support)
        per_stream[str(stream_seed)] = {
            "image_count": len(stream_rows),
            "candidate_count": stream_candidates,
        }

    normal_candidates = [row for row in candidate_buffer if row["image_type"] == "good"]
    anomaly_candidates = [row for row in candidate_buffer if row["image_type"] != "good"]
    eligible_normal = [
        row
        for row in decisions
        if row["image_type"] == "good"
        and row["ms_threshold"] is not None
        and row["support_threshold"] is not None
    ]
    summary = {
        "stage": 2,
        "gate": {
            "condition": "ms_score <= historical q_ms and channel_support >= historical q_support",
            "ms_quantile": ms_quantile,
            "support_quantile": support_quantile,
            "support_method": support_method,
            "threshold_source": "strictly prior values within each stream",
        },
        "candidate_buffer": {
            "candidate_count": len(candidate_buffer),
            "normal_candidate_count": len(normal_candidates),
            "anomaly_candidate_count": len(anomaly_candidates),
            "normal_precision": (
                len(normal_candidates) / len(candidate_buffer) if candidate_buffer else None
            ),
            "eligible_normal_capture_rate": (
                len(normal_candidates) / len(eligible_normal) if eligible_normal else None
            ),
        },
        "expert_admission": {
            "enabled": False,
            "expert_count": 0,
        },
        "per_stream": per_stream,
    }

    fig, axis = plt.subplots(figsize=(7, 5))
    available = [
        row
        for row in decisions
        if row["ms_score"] is not None and row["channel_support"] is not None
    ]
    rejected = [row for row in available if not row["is_candidate"]]
    candidates = [row for row in available if row["is_candidate"]]
    if rejected:
        axis.scatter(
            [row["ms_score"] for row in rejected],
            [row["channel_support"] for row in rejected],
            s=18,
            alpha=0.45,
            label="not selected",
        )
    if candidates:
        axis.scatter(
            [row["ms_score"] for row in candidates],
            [row["channel_support"] for row in candidates],
            s=32,
            alpha=0.85,
            label="Candidate Buffer",
        )
    axis.set_xlabel("online DINO MSM score")
    axis.set_ylabel(f"{support_method} Channel support")
    axis.set_title("Stage 2: Candidate Expert gate")
    axis.grid(alpha=0.25)
    if available:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage2_candidate_gate.png", dpi=200)
    plt.close(fig)

    write_csv(output_dir / "stage2_candidate_decisions.csv", decisions)
    write_csv(output_dir / "stage2_candidate_buffer.csv", candidate_buffer)
    (output_dir / "stage2_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, candidate_buffer


def run_stage3_expert_admission(
    candidate_buffer,
    features,
    output_dir,
    duplicate_similarity=0.98,
    committee_cap=5,
    min_cluster_support=2,
):
    """Admit a buffered candidate only after a similar-candidate cluster confirms it."""
    if not 0.0 <= duplicate_similarity <= 1.0:
        raise ValueError("duplicate_similarity must be within [0, 1]")
    if committee_cap < 1:
        raise ValueError("committee_cap must be at least 1")
    if min_cluster_support < 2:
        raise ValueError("min_cluster_support must be at least 2")

    image_embeddings = torch.stack(
        [feature.float().mean(dim=0) for feature in features]
    )
    image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)
    stream_seeds = sorted({row["stream_seed"] for row in candidate_buffer})
    decisions = []
    admitted_experts = []
    per_stream = {}
    for stream_seed in stream_seeds:
        clusters = []
        committee_size = 0
        stream_rows = sorted(
            (row for row in candidate_buffer if row["stream_seed"] == stream_seed),
            key=lambda row: row["stream_step"],
        )
        for row in stream_rows:
            quality = (
                row["channel_support"] / max(row["support_threshold"], 1e-12)
                - row["ms_score"] / max(row["ms_threshold"], 1e-12)
            )
            gate_passed = bool(
                row["is_candidate"]
                and row["ms_score"] <= row["ms_threshold"]
                and row["channel_support"] >= row["support_threshold"]
            )
            max_similarity = None
            duplicate = False
            cluster_id = None
            cluster_size = 0
            triggered_admission = False
            admitted_image_id = None
            if not gate_passed:
                reason = "gate_failed"
            else:
                if clusters:
                    centroids = torch.stack(
                        [
                            torch.nn.functional.normalize(
                                cluster["embedding_sum"], dim=0
                            )
                            for cluster in clusters
                        ]
                    )
                    similarities = centroids @ image_embeddings[row["image_id"]]
                    max_similarity = float(similarities.max())
                    if max_similarity >= duplicate_similarity:
                        cluster_id = int(similarities.argmax())
                        duplicate = True
                if cluster_id is None:
                    cluster_id = len(clusters)
                    clusters.append(
                        {
                            "embedding_sum": torch.zeros_like(
                                image_embeddings[row["image_id"]]
                            ),
                            "members": [],
                            "expert_admitted": False,
                        }
                    )
                cluster = clusters[cluster_id]
                member = dict(row)
                member["candidate_quality"] = quality
                cluster["members"].append(member)
                cluster["embedding_sum"] += image_embeddings[row["image_id"]]
                cluster_size = len(cluster["members"])
                if cluster["expert_admitted"]:
                    reason = "reinforce_existing_cluster"
                elif cluster_size < min_cluster_support:
                    reason = "pending_cluster_confirmation"
                elif committee_size >= committee_cap:
                    reason = "committee_cap"
                else:
                    representative = max(
                        cluster["members"], key=lambda item: item["candidate_quality"]
                    )
                    member_steps = sorted(
                        member["stream_step"] for member in cluster["members"]
                    )
                    max_support_gap = max(
                        later - earlier
                        for earlier, later in zip(member_steps, member_steps[1:])
                    )
                    expert = dict(representative)
                    expert.update(
                        {
                            "cluster_id": cluster_id,
                            "cluster_support_at_admission": cluster_size,
                            "cluster_max_support_gap": max_support_gap,
                            "admission_trigger_step": row["stream_step"],
                            "expert_index": committee_size,
                            "expert_admitted": True,
                        }
                    )
                    admitted_experts.append(expert)
                    cluster["expert_admitted"] = True
                    committee_size += 1
                    triggered_admission = True
                    admitted_image_id = representative["image_id"]
                    reason = "cluster_confirmed"
            decision = {
                **row,
                "candidate_quality": quality,
                "gate_passed": gate_passed,
                "max_cluster_similarity": max_similarity,
                "duplicate_similarity_threshold": duplicate_similarity,
                "is_duplicate": duplicate,
                "cluster_id": cluster_id,
                "cluster_size_after": cluster_size,
                "min_cluster_support": min_cluster_support,
                "committee_cap": committee_cap,
                "triggered_expert_admission": triggered_admission,
                "admitted_image_id": admitted_image_id,
                "expert_admitted": admitted_image_id == row["image_id"],
                "admission_reason": reason,
                "committee_size_after": committee_size,
            }
            decisions.append(decision)
        per_stream[str(stream_seed)] = {
            "candidate_count": len(stream_rows),
            "cluster_count": len(clusters),
            "confirmed_cluster_count": sum(
                cluster["expert_admitted"] for cluster in clusters
            ),
            "expert_count": committee_size,
        }

    normal_experts = [row for row in admitted_experts if row["image_type"] == "good"]
    anomaly_experts = [row for row in admitted_experts if row["image_type"] != "good"]
    summary = {
        "stage": 3,
        "admission": {
            "condition": "candidate gate passed, similar-candidate cluster confirmed, committee below cap",
            "duplicate_similarity": duplicate_similarity,
            "min_cluster_support": min_cluster_support,
            "committee_cap": committee_cap,
            "embedding": "mean of normalized DINO patch features, then L2 normalized",
            "representative": "highest support-ratio minus MS-ratio within confirmed cluster",
        },
        "committee": {
            "expert_count": len(admitted_experts),
            "normal_expert_count": len(normal_experts),
            "anomaly_expert_count": len(anomaly_experts),
            "normal_precision": (
                len(normal_experts) / len(admitted_experts) if admitted_experts else None
            ),
        },
        "candidate_states": {
            "pending_cluster_confirmation": sum(
                row["admission_reason"] == "pending_cluster_confirmation"
                for row in decisions
            ),
            "reinforce_existing_cluster": sum(
                row["admission_reason"] == "reinforce_existing_cluster"
                for row in decisions
            ),
            "cluster_confirmed": sum(
                row["admission_reason"] == "cluster_confirmed" for row in decisions
            ),
            "committee_cap": sum(
                row["admission_reason"] == "committee_cap" for row in decisions
            ),
            "gate_failed": sum(row["admission_reason"] == "gate_failed" for row in decisions),
        },
        "per_stream": per_stream,
    }

    fig, axis = plt.subplots(figsize=(8, 5))
    for stream_seed in stream_seeds:
        stream_rows = [row for row in decisions if row["stream_seed"] == stream_seed]
        axis.step(
            [row["stream_step"] for row in stream_rows],
            [row["committee_size_after"] for row in stream_rows],
            where="post",
            label=f"seed {stream_seed}",
        )
    axis.axhline(committee_cap, color="black", linestyle="--", linewidth=1, label="cap")
    axis.set_xlabel("stream step of candidate")
    axis.set_ylabel("committee size")
    axis.set_title("Stage 3: Cluster-confirmed Dynamic Expert admission")
    axis.grid(alpha=0.25)
    if stream_seeds:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage3_expert_admission.png", dpi=200)
    plt.close(fig)

    write_csv(output_dir / "stage3_expert_admission_decisions.csv", decisions)
    write_csv(output_dir / "stage3_expert_committee.csv", admitted_experts)
    (output_dir / "stage3_expert_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, admitted_experts


def run_stage4_expert_lifecycle(
    stage1_records,
    admitted_experts,
    features,
    output_dir,
    support_similarity=0.98,
    base_ttl=5,
    max_ttl=20,
    ttl_gap_multiplier=2.0,
    min_committee_size=1,
):
    """Delete experts after an adaptive number of consecutive unsupported images."""
    if not 0.0 <= support_similarity <= 1.0:
        raise ValueError("support_similarity must be within [0, 1]")
    if base_ttl < 1:
        raise ValueError("base_ttl must be at least 1")
    if max_ttl < base_ttl:
        raise ValueError("max_ttl must be greater than or equal to base_ttl")
    if ttl_gap_multiplier < 1.0:
        raise ValueError("ttl_gap_multiplier must be at least 1")
    if min_committee_size < 0:
        raise ValueError("min_committee_size must be non-negative")

    def patience_from_gap(max_support_gap):
        return min(
            max_ttl,
            max(base_ttl, int(math.ceil(max_support_gap * ttl_gap_multiplier))),
        )

    image_embeddings = torch.stack(
        [feature.float().mean(dim=0) for feature in features]
    )
    image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=1)
    local_records = [row for row in stage1_records if row["method"] == "local"]
    stream_seeds = sorted({row["stream_seed"] for row in local_records})
    events = []
    deletions = []
    final_experts = []
    per_stream = {}
    committee_timeline = {}
    for stream_seed in stream_seeds:
        stream_rows = sorted(
            (row for row in local_records if row["stream_seed"] == stream_seed),
            key=lambda row: row["stream_step"],
        )
        scheduled = {}
        for expert in admitted_experts:
            if expert["stream_seed"] == stream_seed:
                scheduled.setdefault(expert["admission_trigger_step"], []).append(expert)
        active = []
        timeline = []
        for row in stream_rows:
            stream_step = row["stream_step"]
            current_embedding = image_embeddings[row["image_id"]]
            survivors = []
            deletion_slots = max(0, len(active) - min_committee_size)
            ordered_active = sorted(
                active,
                key=lambda state: (
                    state["last_supported_step"],
                    state["support_count"],
                ),
            )
            for state in ordered_active:
                similarity = float(
                    image_embeddings[state["image_id"]] @ current_embedding
                )
                supported = similarity >= support_similarity
                state["age"] += 1
                if supported:
                    support_gap = stream_step - state["last_supported_step"]
                    state["max_support_gap"] = max(
                        state["max_support_gap"], support_gap
                    )
                    state["patience"] = patience_from_gap(state["max_support_gap"])
                    state["ttl"] = state["patience"]
                    state["last_supported_step"] = stream_step
                    state["support_count"] += 1
                    event = "supported"
                else:
                    state["ttl"] -= 1
                    event = "ttl_decrement"
                expired = state["ttl"] <= 0
                deleted = expired and deletion_slots > 0
                if deleted:
                    event = "deleted"
                    deletion_slots -= 1
                elif expired:
                    state["ttl"] = 0
                    event = "retained_minimum"
                event_row = {
                    "stream_seed": stream_seed,
                    "stream_step": stream_step,
                    "current_image_id": row["image_id"],
                    "expert_index": state["expert_index"],
                    "expert_image_id": state["image_id"],
                    "expert_image_type": state["image_type"],
                    "support_similarity": similarity,
                    "support_threshold": support_similarity,
                    "supported": supported,
                    "age": state["age"],
                    "ttl": state["ttl"],
                    "patience": state["patience"],
                    "max_support_gap": state["max_support_gap"],
                    "last_supported_step": state["last_supported_step"],
                    "support_count": state["support_count"],
                    "event": event,
                }
                events.append(event_row)
                if deleted:
                    deletion = dict(event_row)
                    deletion["lifetime"] = state["age"]
                    deletions.append(deletion)
                else:
                    survivors.append(state)
            active = survivors

            for expert in scheduled.get(stream_step, []):
                initial_gap = expert["cluster_max_support_gap"]
                initial_patience = patience_from_gap(initial_gap)
                state = {
                    **expert,
                    "age": 0,
                    "ttl": initial_patience,
                    "patience": initial_patience,
                    "max_support_gap": initial_gap,
                    "last_supported_step": stream_step,
                    "support_count": expert["cluster_support_at_admission"],
                }
                active.append(state)
                events.append(
                    {
                        "stream_seed": stream_seed,
                        "stream_step": stream_step,
                        "current_image_id": row["image_id"],
                        "expert_index": state["expert_index"],
                        "expert_image_id": state["image_id"],
                        "expert_image_type": state["image_type"],
                        "support_similarity": None,
                        "support_threshold": support_similarity,
                        "supported": True,
                        "age": 0,
                        "ttl": initial_patience,
                        "patience": initial_patience,
                        "max_support_gap": initial_gap,
                        "last_supported_step": stream_step,
                        "support_count": state["support_count"],
                        "event": "admitted",
                    }
                )
            timeline.append((stream_step, len(active)))

        for state in active:
            final_experts.append(
                {
                    **state,
                    "final_stream_step": stream_rows[-1]["stream_step"],
                }
            )
        committee_timeline[stream_seed] = timeline
        admitted_count = sum(
            expert["stream_seed"] == stream_seed for expert in admitted_experts
        )
        deleted_count = sum(row["stream_seed"] == stream_seed for row in deletions)
        per_stream[str(stream_seed)] = {
            "admitted_count": admitted_count,
            "deleted_count": deleted_count,
            "surviving_count": len(active),
        }

    normal_final = [row for row in final_experts if row["image_type"] == "good"]
    anomaly_final = [row for row in final_experts if row["image_type"] != "good"]
    summary = {
        "stage": 4,
        "lifecycle": {
            "support_similarity": support_similarity,
            "base_ttl": base_ttl,
            "max_ttl": max_ttl,
            "ttl_gap_multiplier": ttl_gap_multiplier,
            "min_committee_size": min_committee_size,
            "patience": "clip(ceil(observed maximum support gap * multiplier), base_ttl, max_ttl)",
            "delete_condition": "adaptive TTL reaches zero and deletion preserves the minimum committee size",
        },
        "experts": {
            "admitted_count": len(admitted_experts),
            "deleted_count": len(deletions),
            "surviving_count": len(final_experts),
            "normal_surviving_count": len(normal_final),
            "anomaly_surviving_count": len(anomaly_final),
            "surviving_normal_precision": (
                len(normal_final) / len(final_experts) if final_experts else None
            ),
            "retained_minimum_events": sum(
                row["event"] == "retained_minimum" for row in events
            ),
        },
        "per_stream": per_stream,
    }

    fig, axis = plt.subplots(figsize=(9, 5))
    for stream_seed, timeline in committee_timeline.items():
        axis.step(
            [item[0] for item in timeline],
            [item[1] for item in timeline],
            where="post",
            label=f"seed {stream_seed}",
        )
    axis.set_xlabel("stream step")
    axis.set_ylabel("active expert count")
    axis.set_title("Stage 4: Expert lifecycle")
    axis.grid(alpha=0.25)
    if committee_timeline:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "stage4_expert_lifecycle.png", dpi=200)
    plt.close(fig)

    write_csv(output_dir / "stage4_expert_lifecycle_events.csv", events)
    write_csv(output_dir / "stage4_expert_deletions.csv", deletions)
    write_csv(output_dir / "stage4_final_committee.csv", final_experts)
    (output_dir / "stage4_expert_lifecycle_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, events, final_experts


def run_stage1_image_support(
    features,
    samples,
    device,
    output_dir,
    ttl=5,
    density_k=5,
    alpha=0.5,
    threshold_quantile=0.7,
    stream_seeds=(0, 1, 2),
    msm_chunk_size=8,
):
    """Measure online Channel support before every Channel memory update."""
    grid_size = int(math.sqrt(features[0].shape[0]))
    records = []
    for stream_seed in stream_seeds:
        order = list(range(len(samples)))
        random.Random(stream_seed).shuffle(order)
        memories = {
            "global": ChannelMemory(
                max_ttl=ttl, density_k=density_k, device=device, position_radius=None
            ),
            "local": ChannelMemory(
                max_ttl=ttl, density_k=density_k, device=device, position_radius=1
            ),
        }
        distance_history = {"global": [], "local": []}
        reference_features = []
        for stream_step, image_id in enumerate(order):
            current = features[image_id]
            ms_score = compute_online_dino_msm_image_score(
                current,
                reference_features,
                device=device,
                reference_chunk_size=msm_chunk_size,
            )
            for method, memory in memories.items():
                mature = memory.weighted_mature_channels()
                threshold = None
                support = None
                current_distances = None
                if mature:
                    current_distances = compute_patch_channel_distances(
                        current,
                        mature,
                        grid_size,
                        device,
                        position_radius=memory.position_radius,
                    )
                    if distance_history[method]:
                        threshold = float(
                            np.quantile(distance_history[method], threshold_quantile)
                        )
                        support = float((current_distances <= threshold).float().mean())
                records.append(
                    {
                        "stream_seed": stream_seed,
                        "stream_step": stream_step,
                        "image_id": image_id,
                        "image_type": samples[image_id]["kind"],
                        "method": method,
                        "reliable_mature_channels": len(mature),
                        "online_threshold": threshold,
                        "channel_support": support,
                        "ms_score": ms_score,
                        "ms_score_source": "single-layer online DINO MSM-style score, image max",
                    }
                )
                if current_distances is not None:
                    finite_distances = current_distances[torch.isfinite(current_distances)]
                    distance_history[method].extend(finite_distances.tolist())
                _, patch_weights = memory.patch_reliability(
                    current, grid_size, reliability_mode="rank"
                )
                soft_weights = alpha + (1.0 - alpha) * patch_weights
                memory.update(
                    current,
                    image_id,
                    patch_densities=None,
                    patch_reliabilities=soft_weights,
                )
            reference_features.append(current)

    summary = {
        "stage": 1,
        "stream_seeds": list(stream_seeds),
        "threshold": {
            "quantile": threshold_quantile,
            "source": "strictly prior finite NN distances within each stream and method",
        },
        "ms_score": {
            "available": True,
            "source": "single-layer DINO MSM-style score against strictly prior stream images",
            "image_reduction": "maximum patch score",
            "unavailable_warmup_images": len(stream_seeds),
        },
        "methods": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for axis, method in zip(axes, ("global", "local")):
        available = [
            row for row in records if row["method"] == method and row["channel_support"] is not None
        ]
        normal = [row["channel_support"] for row in available if row["image_type"] == "good"]
        anomaly = [row["channel_support"] for row in available if row["image_type"] != "good"]
        pairwise = [
            float(normal_value > anomaly_value) + 0.5 * float(normal_value == anomaly_value)
            for normal_value in normal
            for anomaly_value in anomaly
        ]
        summary["methods"][method] = {
            "available_records": len(available),
            "unavailable_warmup_records": len(stream_seeds) * len(samples) - len(available),
            "normal": summarize(normal),
            "anomaly": summarize(anomaly),
            "normal_minus_anomaly_mean": (
                float(np.mean(normal) - np.mean(anomaly)) if normal and anomaly else None
            ),
            "support_separation_auc": float(np.mean(pairwise)) if pairwise else None,
        }
        if normal:
            axis.hist(normal, bins=bins, alpha=0.65, density=True, label="normal")
        if anomaly:
            axis.hist(anomaly, bins=bins, alpha=0.65, density=True, label="anomaly")
        axis.set_title(f"{method.title()} online support")
        axis.set_xlabel("image-level Channel support")
        if normal or anomaly:
            axis.legend()
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(output_dir / "stage1_online_support_distributions.png", dpi=200)
    plt.close(fig)
    write_csv(output_dir / "stage1_online_image_support.csv", records)
    (output_dir / "stage1_online_image_support_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--layer-index", type=int, default=23)
    parser.add_argument("--ttl", type=int, default=5)
    parser.add_argument("--density-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--threshold-quantile", type=float, default=0.7)
    parser.add_argument("--features-cache", type=str, default=None)
    parser.add_argument("--stream-seeds", type=str, default="0,1,2")
    parser.add_argument("--msm-chunk-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--stage1-records", type=str, default=None)
    parser.add_argument("--stage2-candidates", type=str, default=None)
    parser.add_argument("--stage3-experts", type=str, default=None)
    parser.add_argument("--candidate-ms-quantile", type=float, default=0.3)
    parser.add_argument("--candidate-support-quantile", type=float, default=0.7)
    parser.add_argument("--expert-duplicate-similarity", type=float, default=0.98)
    parser.add_argument("--expert-committee-cap", type=int, default=5)
    parser.add_argument("--expert-min-cluster-support", type=int, default=2)
    parser.add_argument("--expert-support-similarity", type=float, default=0.98)
    parser.add_argument("--expert-base-ttl", type=int, default=5)
    parser.add_argument("--expert-max-ttl", type=int, default=20)
    parser.add_argument("--expert-ttl-gap-multiplier", type=float, default=2.0)
    parser.add_argument("--expert-min-committee-size", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root)
    if not samples:
        raise RuntimeError("No bottle test images found")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    if args.stage == 2 and args.stage1_records:
        stage1_records = read_stage1_records(args.stage1_records)
        summary, _ = run_stage2_candidate_buffer(
            stage1_records,
            output_dir,
            ms_quantile=args.candidate_ms_quantile,
            support_quantile=args.candidate_support_quantile,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_cache = (
        Path(args.features_cache)
        if args.features_cache
        else output_dir / "bottle_dino_features.pt"
    )
    if features_cache.exists():
        features = torch.load(features_cache, map_location="cpu", weights_only=False)
        print(f"loaded feature cache: {features_cache}", flush=True)
    else:
        model = torch.hub.load(args.dinov2_repo, "dinov2_vitl14", source="local")
        model.eval().to(device)
        preprocess = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        features = extract_features(samples, model, preprocess, device, args.layer_index)
        torch.save(features, features_cache)

    features = features[: len(samples)]
    if len(features) != len(samples):
        raise RuntimeError("Feature cache contains fewer entries than the selected samples")
    if args.stage == 3 and args.stage2_candidates:
        candidate_buffer = read_stage2_candidates(args.stage2_candidates)
        summary, _ = run_stage3_expert_admission(
            candidate_buffer,
            features,
            output_dir,
            duplicate_similarity=args.expert_duplicate_similarity,
            committee_cap=args.expert_committee_cap,
            min_cluster_support=args.expert_min_cluster_support,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.stage == 4 and args.stage1_records and args.stage3_experts:
        stage1_records = read_stage1_records(args.stage1_records)
        admitted_experts = read_stage3_experts(args.stage3_experts)
        summary, _, _ = run_stage4_expert_lifecycle(
            stage1_records,
            admitted_experts,
            features,
            output_dir,
            support_similarity=args.expert_support_similarity,
            base_ttl=args.expert_base_ttl,
            max_ttl=args.expert_max_ttl,
            ttl_gap_multiplier=args.expert_ttl_gap_multiplier,
            min_committee_size=args.expert_min_committee_size,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    stream_seeds = tuple(int(value.strip()) for value in args.stream_seeds.split(","))
    stage1_summary, stage1_records = run_stage1_image_support(
        features,
        samples,
        device,
        output_dir,
        ttl=args.ttl,
        density_k=args.density_k,
        alpha=args.alpha,
        threshold_quantile=args.threshold_quantile,
        stream_seeds=stream_seeds,
        msm_chunk_size=args.msm_chunk_size,
    )
    if args.stage == 1:
        summary = stage1_summary
    else:
        stage2_summary, candidate_buffer = run_stage2_candidate_buffer(
            stage1_records,
            output_dir,
            ms_quantile=args.candidate_ms_quantile,
            support_quantile=args.candidate_support_quantile,
        )
        if args.stage == 2:
            summary = stage2_summary
        else:
            stage3_summary, admitted_experts = run_stage3_expert_admission(
                candidate_buffer,
                features,
                output_dir,
                duplicate_similarity=args.expert_duplicate_similarity,
                committee_cap=args.expert_committee_cap,
                min_cluster_support=args.expert_min_cluster_support,
            )
            if args.stage == 3:
                summary = stage3_summary
            else:
                summary, _, _ = run_stage4_expert_lifecycle(
                    stage1_records,
                    admitted_experts,
                    features,
                    output_dir,
                    support_similarity=args.expert_support_similarity,
                    base_ttl=args.expert_base_ttl,
                    max_ttl=args.expert_max_ttl,
                    ttl_gap_multiplier=args.expert_ttl_gap_multiplier,
                    min_committee_size=args.expert_min_committee_size,
                )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
