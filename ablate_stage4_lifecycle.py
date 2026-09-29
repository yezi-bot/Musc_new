"""Ablate Stage 4 expert lifecycle refresh signals on MVTec AD bottle."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from validate_density_bottle import ChannelMemory, load_samples, write_csv
from validate_dynamic_ex_msm import (
    compute_patch_channel_distances,
    read_stage1_records,
    read_stage3_experts,
)


VARIANTS = {
    "similarity_ttl": None,
    "channel_ttl": None,
    "hybrid_beta_025": 0.25,
    "hybrid_beta_050": 0.50,
    "hybrid_beta_075": 0.75,
}


def expert_id(expert):
    return f"s{expert['stream_seed']}-e{expert['expert_index']}"


def patience_from_gap(gap, base_ttl, max_ttl, multiplier):
    return min(max_ttl, max(base_ttl, int(math.ceil(gap * multiplier))))


def precompute_online_signals(
    stage1_records,
    admitted_experts,
    features,
    device,
    channel_ttl=5,
    density_k=5,
    alpha=0.5,
    distance_quantile=0.7,
    channel_quantile=0.7,
):
    """Compute expert signals before the current image updates ChannelMemory."""
    grid_size = int(math.sqrt(features[0].shape[0]))
    if grid_size * grid_size != features[0].shape[0]:
        raise ValueError("Patch count must form a square grid")

    embeddings = torch.stack([feature.float().mean(dim=0) for feature in features])
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    local_records = [row for row in stage1_records if row["method"] == "local"]
    stream_seeds = sorted({row["stream_seed"] for row in local_records})
    signals = {}
    stream_rows_by_seed = {}

    for stream_seed in stream_seeds:
        stream_rows = sorted(
            (row for row in local_records if row["stream_seed"] == stream_seed),
            key=lambda row: row["stream_step"],
        )
        stream_rows_by_seed[stream_seed] = stream_rows
        stream_experts = [
            expert for expert in admitted_experts if expert["stream_seed"] == stream_seed
        ]
        memory = ChannelMemory(
            max_ttl=channel_ttl,
            density_k=density_k,
            device=device,
            position_radius=1,
        )
        distance_history = []
        current_support_history = []

        for row in stream_rows:
            step = row["stream_step"]
            current = features[row["image_id"]]
            mature = memory.weighted_mature_channels()
            patch_threshold = (
                float(np.quantile(distance_history, distance_quantile))
                if distance_history
                else None
            )
            channel_threshold = (
                float(np.quantile(current_support_history, channel_quantile))
                if current_support_history
                else None
            )

            for expert in stream_experts:
                if expert["admission_trigger_step"] > step:
                    continue
                similarity = float(
                    embeddings[expert["image_id"]] @ embeddings[row["image_id"]]
                )
                support = None
                if mature and patch_threshold is not None:
                    expert_distances = compute_patch_channel_distances(
                        features[expert["image_id"]],
                        mature,
                        grid_size,
                        device,
                        position_radius=1,
                    )
                    support = float((expert_distances <= patch_threshold).float().mean())
                signals[(expert_id(expert), step)] = {
                    "support_similarity": similarity,
                    "channel_support": support,
                    "channel_threshold": channel_threshold,
                    "patch_distance_threshold": patch_threshold,
                    "reliable_mature_channels": len(mature),
                }

            current_distances = None
            current_support = None
            if mature:
                current_distances = compute_patch_channel_distances(
                    current,
                    mature,
                    grid_size,
                    device,
                    position_radius=1,
                )
                if patch_threshold is not None:
                    current_support = float(
                        (current_distances <= patch_threshold).float().mean()
                    )

            # These histories become visible only to later stream steps.
            if current_distances is not None:
                finite = current_distances[torch.isfinite(current_distances)]
                distance_history.extend(finite.tolist())
            if current_support is not None:
                current_support_history.append(current_support)

            _, patch_weights = memory.patch_reliability(
                current, grid_size, reliability_mode="rank"
            )
            soft_weights = alpha + (1.0 - alpha) * patch_weights
            memory.update(
                current,
                row["image_id"],
                patch_densities=None,
                patch_reliabilities=soft_weights,
            )

    return signals, stream_rows_by_seed


def refresh_decision(variant, signal, similarity_threshold):
    similarity = signal["support_similarity"]
    channel_support = signal["channel_support"]
    channel_threshold = signal["channel_threshold"]
    similarity_refresh = similarity >= similarity_threshold
    channel_refresh = None
    if channel_support is not None and channel_threshold is not None:
        channel_refresh = channel_support >= channel_threshold

    hybrid_score = None
    if variant == "similarity_ttl":
        return similarity_refresh, similarity_refresh, channel_refresh, hybrid_score
    if variant == "channel_ttl":
        return channel_refresh, similarity_refresh, channel_refresh, hybrid_score

    beta = VARIANTS[variant]
    if channel_refresh is None:
        return None, similarity_refresh, channel_refresh, hybrid_score
    sim_margin = similarity / max(similarity_threshold, 1e-12)
    channel_margin = channel_support / max(channel_threshold, 1e-12)
    hybrid_score = beta * channel_margin + (1.0 - beta) * sim_margin
    return hybrid_score >= 1.0, similarity_refresh, channel_refresh, hybrid_score


def run_lifecycle_variant(
    variant,
    committee_floor,
    admitted_experts,
    stream_rows_by_seed,
    signals,
    similarity_threshold=0.98,
    base_ttl=5,
    max_ttl=20,
    ttl_gap_multiplier=2.0,
):
    timeline_rows = []
    final_experts = []
    committee_counts = []

    for stream_seed, stream_rows in stream_rows_by_seed.items():
        scheduled = {}
        for expert in admitted_experts:
            if expert["stream_seed"] == stream_seed:
                scheduled.setdefault(expert["admission_trigger_step"], []).append(expert)
        active = []
        ever_admitted = False

        for row in stream_rows:
            step = row["stream_step"]
            deletion_slots = max(0, len(active) - committee_floor)
            survivors = []
            ordered_active = sorted(
                active,
                key=lambda state: (state["last_supported_step"], state["support_count"]),
            )
            for state in ordered_active:
                signal = signals[(state["expert_id"], step)]
                refresh, similarity_refresh, channel_refresh, hybrid_score = (
                    refresh_decision(variant, signal, similarity_threshold)
                )
                ttl_before = state["ttl"]
                state["age"] += 1
                if refresh is None:
                    event = "signal_unavailable"
                elif refresh:
                    support_gap = step - state["last_supported_step"]
                    state["max_support_gap"] = max(
                        state["max_support_gap"], support_gap
                    )
                    state["patience"] = patience_from_gap(
                        state["max_support_gap"],
                        base_ttl,
                        max_ttl,
                        ttl_gap_multiplier,
                    )
                    state["ttl"] = state["patience"]
                    state["last_supported_step"] = step
                    state["support_count"] += 1
                    event = "ttl_refreshed"
                else:
                    state["ttl"] -= 1
                    event = "ttl_decrement"

                expired = state["ttl"] <= 0
                deleted = expired and deletion_slots > 0
                if deleted:
                    state["ttl"] = 0
                    event = "deleted"
                    deletion_slots -= 1
                elif expired:
                    state["ttl"] = 0
                    event = "retained_minimum"

                timeline_rows.append(
                    {
                        "step": step,
                        "variant": variant,
                        "committee_floor": committee_floor,
                        "expert_id": state["expert_id"],
                        "stream_seed": stream_seed,
                        "expert_image_id": state["image_id"],
                        "gt_debug": state["image_type"],
                        "support_similarity": signal["support_similarity"],
                        "channel_support": signal["channel_support"],
                        "similarity_threshold": similarity_threshold,
                        "channel_threshold": signal["channel_threshold"],
                        "patch_distance_threshold": signal[
                            "patch_distance_threshold"
                        ],
                        "hybrid_score": hybrid_score,
                        "ttl_before": ttl_before,
                        "ttl_after": state["ttl"],
                        "patience": state["patience"],
                        "age": state["age"],
                        "similarity_refresh": similarity_refresh,
                        "channel_refresh": channel_refresh,
                        "alive": not deleted,
                        "deleted_this_step": deleted,
                        "event": event,
                    }
                )
                if not deleted:
                    survivors.append(state)
            active = survivors

            for expert in scheduled.get(step, []):
                gap = expert["cluster_max_support_gap"]
                initial_ttl = patience_from_gap(
                    gap, base_ttl, max_ttl, ttl_gap_multiplier
                )
                state = {
                    **expert,
                    "expert_id": expert_id(expert),
                    "age": 0,
                    "ttl": initial_ttl,
                    "patience": initial_ttl,
                    "max_support_gap": gap,
                    "last_supported_step": step,
                    "support_count": expert["cluster_support_at_admission"],
                }
                active.append(state)
                ever_admitted = True
                signal = signals[(state["expert_id"], step)]
                timeline_rows.append(
                    {
                        "step": step,
                        "variant": variant,
                        "committee_floor": committee_floor,
                        "expert_id": state["expert_id"],
                        "stream_seed": stream_seed,
                        "expert_image_id": state["image_id"],
                        "gt_debug": state["image_type"],
                        "support_similarity": signal["support_similarity"],
                        "channel_support": signal["channel_support"],
                        "similarity_threshold": similarity_threshold,
                        "channel_threshold": signal["channel_threshold"],
                        "patch_distance_threshold": signal[
                            "patch_distance_threshold"
                        ],
                        "hybrid_score": None,
                        "ttl_before": initial_ttl,
                        "ttl_after": initial_ttl,
                        "patience": initial_ttl,
                        "age": 0,
                        "similarity_refresh": False,
                        "channel_refresh": False,
                        "alive": True,
                        "deleted_this_step": False,
                        "event": "admitted",
                    }
                )

            committee_counts.append(
                {
                    "variant": variant,
                    "committee_floor": committee_floor,
                    "stream_seed": stream_seed,
                    "step": step,
                    "active_count": len(active),
                    "empty_after_admission": bool(ever_admitted and not active),
                }
            )

        final_experts.extend(active)

    return timeline_rows, final_experts, committee_counts


def summarize_variant(variant, floor, admitted_experts, timeline, final_experts, counts):
    admitted_normal = sum(e["image_type"] == "good" for e in admitted_experts)
    admitted_anomaly = len(admitted_experts) - admitted_normal
    final_normal = sum(e["image_type"] == "good" for e in final_experts)
    final_anomaly = len(final_experts) - final_normal
    deleted = [row for row in timeline if row["deleted_this_step"]]
    return {
        "variant": variant,
        "committee_floor": floor,
        "admitted_count": len(admitted_experts),
        "admitted_normal_count": admitted_normal,
        "admitted_anomaly_count": admitted_anomaly,
        "deleted_count": len(deleted),
        "normal_deleted_count": admitted_normal - final_normal,
        "anomaly_deleted_count": admitted_anomaly - final_anomaly,
        "normal_false_deletion_rate": (
            (admitted_normal - final_normal) / admitted_normal
            if admitted_normal
            else None
        ),
        "anomaly_deletion_rate": (
            (admitted_anomaly - final_anomaly) / admitted_anomaly
            if admitted_anomaly
            else None
        ),
        "surviving_count": len(final_experts),
        "normal_surviving_count": final_normal,
        "anomaly_surviving_count": final_anomaly,
        "surviving_normal_precision": (
            final_normal / len(final_experts) if final_experts else None
        ),
        "empty_committee_steps": sum(row["empty_after_admission"] for row in counts),
        "ttl_refresh_count": sum(row["event"] == "ttl_refreshed" for row in timeline),
        "signal_unavailable_count": sum(
            row["event"] == "signal_unavailable" for row in timeline
        ),
        "retained_minimum_count": sum(
            row["event"] == "retained_minimum" for row in timeline
        ),
    }


def read_deletion_signature(path):
    if path is None or not Path(path).exists():
        return None
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(
        (
            int(row["stream_seed"]),
            int(row["expert_index"]),
            int(row["stream_step"]),
        )
        for row in rows
        if row["event"] == "deleted"
    )


def plot_signal_trajectories(path, admitted_experts, signals, timeline):
    experts = sorted(admitted_experts, key=lambda e: (e["stream_seed"], e["expert_index"]))
    fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharey=True)
    colors = dict(zip(VARIANTS, plt.cm.tab10.colors[: len(VARIANTS)]))
    primary = [row for row in timeline if row["committee_floor"] == 0]
    for axis, expert in zip(axes.flat, experts):
        eid = expert_id(expert)
        points = sorted(
            ((step, value) for (key, step), value in signals.items() if key == eid),
            key=lambda item: item[0],
        )
        steps = [item[0] for item in points]
        sim = [item[1]["support_similarity"] for item in points]
        channel = [item[1]["channel_support"] for item in points]
        thresholds = [item[1]["channel_threshold"] for item in points]
        axis.plot(steps, sim, label="S_sim", color="tab:blue", linewidth=1.2)
        axis.plot(steps, channel, label="S_ch", color="tab:orange", linewidth=1.2)
        axis.plot(
            steps,
            thresholds,
            label="channel q0.7",
            color="tab:orange",
            linestyle="--",
            linewidth=1,
        )
        axis.axhline(0.98, color="tab:blue", linestyle="--", linewidth=1)
        axis.axvline(
            expert["admission_trigger_step"], color="black", linestyle=":", linewidth=1
        )
        rows = [row for row in primary if row["expert_id"] == eid]
        for offset, variant in enumerate(VARIANTS):
            variant_rows = [row for row in rows if row["variant"] == variant]
            refresh_steps = [
                row["step"] for row in variant_rows if row["event"] == "ttl_refreshed"
            ]
            delete_steps = [
                row["step"] for row in variant_rows if row["deleted_this_step"]
            ]
            if refresh_steps:
                axis.scatter(
                    refresh_steps,
                    [0.025 + 0.018 * offset] * len(refresh_steps),
                    marker="|",
                    color=colors[variant],
                    s=35,
                )
            if delete_steps:
                axis.scatter(
                    delete_steps,
                    [0.025 + 0.018 * offset] * len(delete_steps),
                    marker="x",
                    color=colors[variant],
                    s=45,
                )
        axis.set_title(f"{eid}: {expert['image_type']}")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("stream step")
    for axis in axes[:, 0]:
        axis.set_ylabel("support / event markers")
    signal_handles, signal_labels = axes.flat[0].get_legend_handles_labels()
    event_handles = [
        Line2D([0], [0], color="black", linestyle=":", label="admission"),
        Line2D([0], [0], color="black", marker="|", linestyle="None", label="refresh"),
        Line2D([0], [0], color="black", marker="x", linestyle="None", label="deletion"),
    ]
    variant_handles = [
        Line2D([0], [0], color=colors[name], linewidth=3, label=name)
        for name in VARIANTS
    ]
    fig.legend(
        signal_handles + event_handles + variant_handles,
        signal_labels + ["admission", "refresh", "deletion"] + list(VARIANTS),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=6,
        fontsize=8,
    )
    fig.suptitle(
        "Stage 4 expert signals (bottom markers use committee_floor=0)", y=0.998
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_focus_experts(path, focus_ids, admitted_experts, signals, timeline):
    experts_by_id = {expert_id(expert): expert for expert in admitted_experts}
    focus_ids = [eid for eid in focus_ids if eid in experts_by_id]
    if not focus_ids:
        return
    fig, axes = plt.subplots(len(focus_ids), 1, figsize=(12, 3.4 * len(focus_ids)))
    axes = np.atleast_1d(axes)
    for axis, eid in zip(axes, focus_ids):
        points = sorted(
            ((step, value) for (key, step), value in signals.items() if key == eid),
            key=lambda item: item[0],
        )
        steps = [item[0] for item in points]
        axis.plot(
            steps,
            [item[1]["support_similarity"] for item in points],
            label="S_sim",
        )
        axis.plot(
            steps,
            [item[1]["channel_support"] for item in points],
            label="S_ch",
        )
        axis.plot(
            steps,
            [item[1]["channel_threshold"] for item in points],
            linestyle="--",
            label="channel q0.7",
        )
        axis.axhline(0.98, color="tab:blue", linestyle="--", linewidth=1)
        for row in timeline:
            if (
                row["committee_floor"] == 0
                and row["expert_id"] == eid
                and row["deleted_this_step"]
            ):
                axis.axvline(row["step"], color="tab:red", alpha=0.35)
        expert = experts_by_id[eid]
        axis.set_title(f"{eid}: {expert['image_type']}")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.2)
        axis.legend(ncol=3)
    axes[-1].set_xlabel("stream step")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_comparison(path, summaries):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    variants = list(VARIANTS)
    x = np.arange(len(variants))
    width = 0.36
    for offset, floor in ((-width / 2, 0), (width / 2, 1)):
        rows = {
            row["variant"]: row
            for row in summaries
            if row["committee_floor"] == floor
        }
        axes[0, 0].bar(
            x + offset,
            [rows[v]["normal_false_deletion_rate"] for v in variants],
            width,
            label=f"floor={floor}",
        )
        axes[0, 1].bar(
            x + offset,
            [rows[v]["anomaly_deletion_rate"] for v in variants],
            width,
            label=f"floor={floor}",
        )
        axes[1, 0].bar(
            x + offset,
            [rows[v]["normal_surviving_count"] for v in variants],
            width,
            label=f"normal floor={floor}",
        )
        axes[1, 0].bar(
            x + offset,
            [rows[v]["anomaly_surviving_count"] for v in variants],
            width,
            bottom=[rows[v]["normal_surviving_count"] for v in variants],
            alpha=0.45,
            label=f"anomaly floor={floor}",
        )
        axes[1, 1].bar(
            x + offset,
            [rows[v]["empty_committee_steps"] for v in variants],
            width,
            label=f"floor={floor}",
        )
    titles = (
        "Normal false-deletion rate",
        "Anomaly deletion rate",
        "Final committee composition",
        "Empty committee steps after admission",
    )
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xticks(x, variants, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validate_outputs(admitted_experts, timeline, summaries, baseline_signature):
    ids = [expert_id(expert) for expert in admitted_experts]
    errors = []
    if len(ids) != len(set(ids)):
        errors.append("expert_id is not unique")
    if len(admitted_experts) != 8:
        errors.append(f"expected 8 admitted experts, got {len(admitted_experts)}")
    for row in timeline:
        if row["deleted_this_step"] and (row["ttl_after"] != 0 or row["alive"]):
            errors.append(f"invalid deletion row: {row['expert_id']} step {row['step']}")
            break
        if row["event"] != "admitted" and row["variant"] == "similarity_ttl":
            expected = row["support_similarity"] >= row["similarity_threshold"]
            if row["event"] == "ttl_refreshed" and not expected:
                errors.append("similarity refresh violated threshold")
                break
        if row["event"] != "admitted" and row["variant"] == "channel_ttl":
            available = row["channel_support"] is not None and row["channel_threshold"] is not None
            if row["event"] == "ttl_refreshed" and (
                not available or row["channel_support"] < row["channel_threshold"]
            ):
                errors.append("channel refresh violated threshold")
                break
        if row["event"] != "admitted" and row["variant"].startswith("hybrid_"):
            available = row["channel_support"] is not None and row["channel_threshold"] is not None
            if not available:
                if row["hybrid_score"] is not None or row["event"] != "signal_unavailable":
                    errors.append("hybrid lifecycle used unavailable Channel evidence")
                    break
            else:
                beta = VARIANTS[row["variant"]]
                expected_score = beta * (
                    row["channel_support"] / max(row["channel_threshold"], 1e-12)
                ) + (1.0 - beta) * (
                    row["support_similarity"] / row["similarity_threshold"]
                )
                if not math.isclose(row["hybrid_score"], expected_score, rel_tol=1e-9):
                    errors.append("hybrid score violated its normalization formula")
                    break
                expected_refresh = expected_score >= 1.0
                if (row["event"] == "ttl_refreshed") != expected_refresh:
                    errors.append("hybrid refresh violated score threshold")
                    break
    floor_one = [row for row in summaries if row["committee_floor"] == 1]
    if any(row["empty_committee_steps"] for row in floor_one):
        errors.append("committee_floor=1 produced an empty committee")

    actual_signature = sorted(
        (row["stream_seed"], int(row["expert_id"].split("-e")[1]), row["step"])
        for row in timeline
        if row["variant"] == "similarity_ttl"
        and row["committee_floor"] == 1
        and row["deleted_this_step"]
    )
    reproduction = {
        "reference_available": baseline_signature is not None,
        "reference_deletions": baseline_signature,
        "ablation_deletions": actual_signature,
        "deletions_match": (
            actual_signature == baseline_signature if baseline_signature is not None else None
        ),
    }
    if baseline_signature is not None and not reproduction["deletions_match"]:
        errors.append("similarity_ttl floor=1 did not reproduce baseline deletions")
    return errors, reproduction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features-cache", required=True)
    parser.add_argument("--stage1-records", required=True)
    parser.add_argument("--stage3-experts", required=True)
    parser.add_argument("--baseline-events", default=None)
    parser.add_argument("--channel-ttl", type=int, default=5)
    parser.add_argument("--density-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--distance-quantile", type=float, default=0.7)
    parser.add_argument("--channel-quantile", type=float, default=0.7)
    parser.add_argument("--similarity-threshold", type=float, default=0.98)
    parser.add_argument("--expert-base-ttl", type=int, default=5)
    parser.add_argument("--expert-max-ttl", type=int, default=20)
    parser.add_argument("--expert-ttl-gap-multiplier", type=float, default=2.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data_root)
    features = torch.load(args.features_cache, map_location="cpu", weights_only=False)
    if not samples or len(features) != len(samples):
        raise RuntimeError(
            f"Sample/feature mismatch: {len(samples)} samples, {len(features)} features"
        )
    stage1_records = read_stage1_records(args.stage1_records)
    admitted_experts = read_stage3_experts(args.stage3_experts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signals, stream_rows_by_seed = precompute_online_signals(
        stage1_records,
        admitted_experts,
        features,
        device,
        channel_ttl=args.channel_ttl,
        density_k=args.density_k,
        alpha=args.alpha,
        distance_quantile=args.distance_quantile,
        channel_quantile=args.channel_quantile,
    )

    all_timeline = []
    all_summaries = []
    for floor in (0, 1):
        for variant in VARIANTS:
            timeline, final_experts, counts = run_lifecycle_variant(
                variant,
                floor,
                admitted_experts,
                stream_rows_by_seed,
                signals,
                similarity_threshold=args.similarity_threshold,
                base_ttl=args.expert_base_ttl,
                max_ttl=args.expert_max_ttl,
                ttl_gap_multiplier=args.expert_ttl_gap_multiplier,
            )
            all_timeline.extend(timeline)
            all_summaries.append(
                summarize_variant(
                    variant,
                    floor,
                    admitted_experts,
                    timeline,
                    final_experts,
                    counts,
                )
            )

    baseline_path = args.baseline_events
    if baseline_path is None:
        candidate = Path(args.stage3_experts).parent / "stage4_expert_lifecycle_events.csv"
        baseline_path = candidate if candidate.exists() else None
    errors, reproduction = validate_outputs(
        admitted_experts,
        all_timeline,
        all_summaries,
        read_deletion_signature(baseline_path),
    )

    baseline_deleted_ids = sorted(
        {
            row["expert_id"]
            for row in all_timeline
            if row["variant"] == "similarity_ttl"
            and row["committee_floor"] == 1
            and row["deleted_this_step"]
        }
    )
    write_csv(output_dir / "stage4_ablation_timeline.csv", all_timeline)
    write_csv(output_dir / "stage4_ablation_summary.csv", all_summaries)
    plot_signal_trajectories(
        output_dir / "stage4_signal_trajectories.png",
        admitted_experts,
        signals,
        all_timeline,
    )
    plot_focus_experts(
        output_dir / "stage4_focus_deleted_experts.png",
        baseline_deleted_ids,
        admitted_experts,
        signals,
        all_timeline,
    )
    plot_comparison(output_dir / "stage4_ablation_comparison.png", all_summaries)

    result = {
        "experiment": "Stage 4 lifecycle refresh-signal ablation",
        "device": str(device),
        "expert_count": len(admitted_experts),
        "variants": VARIANTS,
        "committee_floors": [0, 1],
        "channel_support": {
            "position_radius": 1,
            "maturity": "effective_span >= 3",
            "patch_distance_quantile": args.distance_quantile,
            "channel_support_quantile": args.channel_quantile,
            "history": "strictly prior observations in the same stream",
            "timing": "expert support and decisions precede current-image ChannelMemory update",
        },
        "baseline_reproduction": reproduction,
        "validation_errors": errors,
        "results": all_summaries,
    }
    (output_dir / "stage4_ablation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError("Ablation validation failed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
