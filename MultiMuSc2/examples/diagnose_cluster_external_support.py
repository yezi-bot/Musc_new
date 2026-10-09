import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from MultiMuSc2.examples.ablate_patch_lof_soft_support import (
    extract_seed_stream,
    validate_config,
)
from MultiMuSc2.examples.ablate_patch_lof_soft_support_v2 import (
    SoftLofLifecycleStateV2,
    config_for_variant,
)
from MultiMuSc2.examples.run_dynamic_dino import (
    git_commit,
    load_dinov2,
    sha256_file,
    write_csv,
)
from MultiMuSc2.models.modules._CHANNEL import ChannelMemory


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose candidate-cluster self-support with provenance-aware "
            "leave-cluster-out Channel support."
        )
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "dynamic_dino.yaml",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--external-support-quantile",
        type=float,
        default=0.7,
    )
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


class ProvenanceChannelMemory(ChannelMemory):
    def update(
        self,
        features,
        image_id,
        grid_size,
        patch_reliabilities,
    ):
        reliabilities = torch.as_tensor(
            patch_reliabilities,
            dtype=torch.float32,
        ).cpu()
        existing_channels = set(self.channels)
        super().update(
            features=features,
            image_id=image_id,
            grid_size=grid_size,
            patch_reliabilities=reliabilities,
        )

        for channel in self.channels:
            if channel not in existing_channels:
                patch_id = int(channel.seed_patch_id)
                channel.source_observations = [
                    (
                        int(channel.seed_image_id),
                        patch_id,
                        float(reliabilities[patch_id]),
                    )
                ]
            elif channel.latest_image_id == int(image_id):
                patch_id = int(channel.latest_patch_id)
                channel.source_observations.append(
                    (
                        int(image_id),
                        patch_id,
                        float(reliabilities[patch_id]),
                    )
                )

            provenance_span = sum(
                observation[2]
                for observation in channel.source_observations
            )
            if abs(provenance_span - channel.effective_span) > 1.0e-5:
                raise RuntimeError(
                    "Channel provenance span does not match effective span"
                )

    def patch_to_external_mature_distances(
        self,
        features,
        grid_size,
        excluded_image_ids,
        feature_bank,
    ):
        self._validate_features(features, grid_size)
        excluded = {int(image_id) for image_id in excluded_image_ids}
        external_channels = []

        for channel in self.channels:
            observations = [
                observation
                for observation in channel.source_observations
                if observation[0] not in excluded
            ]
            external_span = sum(
                observation[2] for observation in observations
            )
            if external_span < self.mature_span:
                continue
            seed_image_id, seed_patch_id, _ = observations[0]
            _, latest_patch_id, _ = observations[-1]
            external_channels.append(
                (
                    feature_bank[seed_image_id][seed_patch_id],
                    latest_patch_id,
                    external_span,
                )
            )

        if not external_channels:
            return None, 0

        current = features.detach().float().to(self.device)
        seeds = torch.stack(
            [channel[0] for channel in external_channels]
        ).float().to(self.device)
        distances = torch.cdist(current, seeds)
        valid = self._position_mask(
            current.shape[0],
            [channel[1] for channel in external_channels],
            grid_size,
        )
        distances = distances.masked_fill(~valid, float("inf"))
        return distances.amin(dim=1).cpu(), len(external_channels)


def support_from_distances(distances, threshold):
    if distances is None or threshold is None:
        return None
    return float((distances <= float(threshold)).float().mean())


def replay_baseline(stream, config, device):
    state = SoftLofLifecycleStateV2(config, device, "baseline")
    for item in stream:
        state.process(item["features"], item["lof_scores"])
    return state.timeline


def diagnostic_replay(
    stream,
    baseline_timeline,
    config,
    seed,
    device,
    external_support_quantile,
):
    committee = config["models"]["dynamic_committee"]
    memory = ProvenanceChannelMemory(
        max_ttl=int(committee.get("channel_ttl", 5)),
        mature_span=float(committee.get("mature_span", 3.0)),
        density_k=int(committee.get("density_k", 5)),
        device=device,
        position_radius=int(committee.get("position_radius", 1)),
    )
    reliability_alpha = float(
        committee.get("reliability_alpha", 0.5)
    )
    feature_bank = []
    cluster_members = defaultdict(list)
    external_support_history = []
    diagnostic_rows = []

    for item, baseline in zip(stream, baseline_timeline):
        step = int(item["step"])
        features = item["features"]
        patch_count = int(features.shape[0])
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError("patch count must form a square grid")

        distance_threshold = baseline["channel_distance_threshold"]
        raw_distances = memory.patch_to_mature_distances(
            features,
            grid_size,
        )
        replayed_support = support_from_distances(
            raw_distances,
            distance_threshold,
        )
        baseline_support = baseline["raw_channel_support"]
        if replayed_support is None or baseline_support is None:
            if replayed_support != baseline_support:
                raise RuntimeError(
                    f"baseline support availability mismatch at step {step}"
                )
        elif abs(replayed_support - baseline_support) > 1.0e-6:
            raise RuntimeError(
                f"baseline support mismatch at step {step}"
            )

        cluster_id = baseline["candidate_cluster_id"]
        is_candidate = bool(baseline["is_candidate"])
        if is_candidate and cluster_id is not None:
            cluster_id = int(cluster_id)
            excluded_ids = list(cluster_members[cluster_id])
            external_distances, external_channel_count = (
                memory.patch_to_external_mature_distances(
                    features=features,
                    grid_size=grid_size,
                    excluded_image_ids=excluded_ids,
                    feature_bank=feature_bank,
                )
            )
            external_support = support_from_distances(
                external_distances,
                distance_threshold,
            )
            external_threshold = (
                float(
                    np.quantile(
                        external_support_history,
                        external_support_quantile,
                    )
                )
                if external_support_history
                else None
            )
            external_decision_available = bool(
                external_support is not None
                and external_threshold is not None
            )
            external_pass = (
                external_support >= external_threshold
                if external_decision_available
                else None
            )
            support_drop = (
                baseline_support - external_support
                if baseline_support is not None
                and external_support is not None
                else None
            )
            support_ratio = (
                external_support / max(baseline_support, 1.0e-12)
                if baseline_support is not None
                and external_support is not None
                else None
            )
            diagnostic_rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "cluster_id": cluster_id,
                    "cluster_member_ids_before": json.dumps(
                        excluded_ids
                    ),
                    "cluster_support_after_current": len(excluded_ids) + 1,
                    "original_channel_support": baseline_support,
                    "external_channel_support": external_support,
                    "support_drop": support_drop,
                    "external_to_original_ratio": support_ratio,
                    "external_support_threshold": external_threshold,
                    "external_decision_available": (
                        external_decision_available
                    ),
                    "hypothetical_external_pass": external_pass,
                    "mature_channel_count_before": len(
                        memory.mature_channels()
                    ),
                    "external_mature_channel_count": (
                        external_channel_count
                    ),
                    "admitted_expert_id": baseline[
                        "admitted_expert_id"
                    ],
                }
            )
            if external_support is not None:
                external_support_history.append(external_support)

        _, reliability = memory.patch_reliability(features, grid_size)
        span_weights = (
            reliability_alpha
            + (1.0 - reliability_alpha) * reliability
        )
        memory.update(
            features=features,
            image_id=step,
            grid_size=grid_size,
            patch_reliabilities=span_weights,
        )
        if len(memory.mature_channels()) != baseline[
            "mature_channel_count_after"
        ]:
            raise RuntimeError(
                f"mature Channel replay mismatch at step {step}"
            )
        feature_bank.append(features.detach().cpu().clone())

        if is_candidate and cluster_id is not None:
            cluster_members[int(cluster_id)].append(step)

    for row in diagnostic_rows:
        anomaly_type = stream[row["step"]]["anomaly_type"]
        row["anomaly_type_debug"] = anomaly_type
        row["is_anomaly_debug"] = int(anomaly_type != "good")
        row["image_path"] = stream[row["step"]]["image_path"]
    return diagnostic_rows


def mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_rows(rows):
    admitted = [
        row for row in rows if row["admitted_expert_id"] is not None
    ]
    normal = [row for row in admitted if not row["is_anomaly_debug"]]
    abnormal = [row for row in admitted if row["is_anomaly_debug"]]
    return {
        "candidate_count": len(rows),
        "admission_count": len(admitted),
        "normal_admission_count": len(normal),
        "abnormal_admission_count": len(abnormal),
        "normal_admission_external_support_mean": mean_or_none(
            [row["external_channel_support"] for row in normal]
        ),
        "abnormal_admission_external_support_mean": mean_or_none(
            [row["external_channel_support"] for row in abnormal]
        ),
        "normal_admission_support_drop_mean": mean_or_none(
            [row["support_drop"] for row in normal]
        ),
        "abnormal_admission_support_drop_mean": mean_or_none(
            [row["support_drop"] for row in abnormal]
        ),
        "normal_hypothetical_rejections": sum(
            row["hypothetical_external_pass"] is False for row in normal
        ),
        "abnormal_hypothetical_rejections": sum(
            row["hypothetical_external_pass"] is False for row in abnormal
        ),
        "normal_external_decision_unavailable": sum(
            not row["external_decision_available"] for row in normal
        ),
        "abnormal_external_decision_unavailable": sum(
            not row["external_decision_available"] for row in abnormal
        ),
    }


def main():
    args = parse_args()
    if not 0.0 < args.external_support_quantile < 1.0:
        raise ValueError(
            "external_support_quantile must be between 0 and 1"
        )
    config_path = Path(args.config).resolve()
    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(base_config)
    config = config_for_variant(base_config, "baseline")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(
        args.dinov2_repo,
        args.checkpoint,
        device,
    )

    all_rows = []
    seed_summaries = {}
    runtimes = {}
    for seed in args.seeds:
        started = time.perf_counter()
        stream = extract_seed_stream(
            base_config,
            args,
            seed,
            model,
            device,
        )
        baseline_timeline = replay_baseline(stream, config, device)
        rows = diagnostic_replay(
            stream=stream,
            baseline_timeline=baseline_timeline,
            config=config,
            seed=seed,
            device=device,
            external_support_quantile=(
                args.external_support_quantile
            ),
        )
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        write_csv(seed_dir / "cluster_external_support.csv", rows)
        (seed_dir / "cluster_external_support.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        seed_summaries[str(seed)] = summarize_rows(rows)
        runtimes[str(seed)] = time.perf_counter() - started
        all_rows.extend(rows)
        del stream, baseline_timeline
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "cluster_external_support_all.csv", all_rows)
    summary = {
        "git_commit": git_commit(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "seeds": args.seeds,
        "max_samples": args.max_samples,
        "external_support_quantile": args.external_support_quantile,
        "method": (
            "fixed-assignment Channel provenance; exclude prior candidate "
            "members of the same cluster from effective span and seed choice"
        ),
        "causal_order": (
            "external support and threshold are computed before current "
            "Channel update and before current support enters history"
        ),
        "gt_usage": (
            "GT is attached only after diagnostic replay for offline labels"
        ),
        "seed_summaries": seed_summaries,
        "overall": summarize_rows(all_rows),
        "runtime_seconds": runtimes,
    }
    (output_dir / "cluster_external_support_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
