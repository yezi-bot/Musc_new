import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.analyze_cross_category_expert_contamination import (
    CrossCategoryFeatureExtractor,
    dino_patch_score,
    interpolate_patch_score,
    selected_stream,
)
from examples.run_dynamic_dino import load_dinov2, seed_everything
from models.backbone.open_clip.transform import image_transform
from models.modules._CHANNEL import ChannelMemory
from models.modules._DYNAMIC_EXPERT import DynamicExpertManager
from models.modules._DYNAMIC_PIPELINE import _channel_support, _image_msm_profile


MS_SHORT_WINDOWS = (8, 16, 32)
DISCRETE_TIMELINE_FIELDS = (
    "active_experts_before",
    "provisional_expert_before",
    "fuser_training_members_before",
    "candidate_route",
    "admission_route",
    "admitted_expert_id",
    "deleted_expert_ids",
    "active_experts_after",
    "active_expert_ids_after",
    "channel_count_after",
    "mature_channel_count_after",
)
DISCRETE_SCORING_FIELDS = (
    "effective_mode",
    "active_expert_ids",
    "expert_image_ids",
    "provisional_expert_image_id",
    "provisional_expert_image_ids",
    "provisional_state",
    "fallback_reason",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证逐图在线执行与原跨类别批量因果模拟是否等价。"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--dinov2-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sequence", default="bottle->cable")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segment-samples", type=int, default=40)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "dynamic_dino_quarantine",
            "dynamic_dino_channel_gate_no_fallback",
        ],
    )
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--float-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def manager_kwargs(variant):
    admission_mode = {
        "dynamic_dino_quarantine": "quarantine",
        "dynamic_dino_channel_gate_no_fallback": "channel_gate",
    }.get(variant)
    if admission_mode is None:
        raise ValueError(f"不支持的验证变体: {variant}")
    return {
        "ms_quantile": 0.3,
        "support_quantile": 0.7,
        "duplicate_similarity": 0.98,
        "min_cluster_support": 2,
        "committee_cap": 5,
        "base_ttl": 5,
        "max_ttl": 20,
        "ttl_gap_multiplier": 2.0,
        "representative_mode": "historical_quality",
        "admission_ttl_mode": "historical_gap",
        "candidate_mode": "dual_path",
        "novel_ms_quantile": 0.9,
        "novel_support_quantile": 0.3,
        "novel_provisional_min": 0.05,
        "novel_admission_mode": admission_mode,
        "novel_quarantine_steps": 8,
        "novel_quarantine_tail": 4,
        "novel_ratio_threshold": 0.95,
        "novel_provisional_weight": 0.25,
        "novel_confirmed_weight": 0.5,
        "novel_confirmed_ttl": 16,
        "novel_confirmed_members": 1,
        "novel_memory_window": 16,
        "novel_memory_keep_fraction": 0.5,
    }


class OnlineTimelineState:
    def __init__(self, device, variant):
        self.device = torch.device(device)
        self.memory = ChannelMemory(
            max_ttl=5,
            mature_span=3.0,
            density_k=5,
            device=self.device,
            position_radius=1,
        )
        self.manager = DynamicExpertManager(**manager_kwargs(variant))
        self.historical_features = []
        self.distance_history = []

    def prepare(self, current, step):
        grid_size = math.isqrt(current.shape[0])
        if grid_size * grid_size != current.shape[0]:
            raise ValueError("patch 数量必须构成正方形网格")
        active_before = self.manager.active_experts_before_step()
        provisional_before = self.manager.provisional_expert_before_step()
        training_members = self.manager.fuser_training_members(step)
        ms_score, ms_short_scores = _image_msm_profile(
            current,
            self.historical_features,
            short_windows=MS_SHORT_WINDOWS,
        )
        ms_short = ms_short_scores[str(MS_SHORT_WINDOWS[0])]
        ms_short_ratio = (
            ms_short / ms_score
            if ms_score is not None and ms_score > 0.0
            else None
        )
        distance_threshold = (
            float(np.quantile(self.distance_history, 0.7))
            if self.distance_history
            else None
        )
        channel_support, current_distances = _channel_support(
            self.memory,
            current,
            grid_size,
            distance_threshold,
        )
        association_profile = self.memory.association_profile(
            current,
            grid_size,
            current_image_id=step,
        )
        channel_ids, _ = self.memory.patch_associations(current, grid_size)
        _, reliability = self.memory.patch_reliability(current, grid_size)
        memory_patch_ids = torch.nonzero(
            (channel_ids >= 0) & (reliability >= 0.5), as_tuple=False
        ).flatten().tolist()
        expert_supports = {}
        for expert in active_before:
            expert_features = self.historical_features[expert["image_id"]].to(
                self.device
            )
            support, _ = _channel_support(
                self.memory,
                expert_features,
                grid_size,
                distance_threshold,
            )
            expert_supports[expert["expert_id"]] = support
        return {
            "step": step,
            "current": current,
            "grid_size": grid_size,
            "active_experts_before": active_before,
            "provisional_expert_before": provisional_before,
            "fuser_training_members_before": training_members,
            "ms_score": ms_score,
            "ms_short_scores": ms_short_scores,
            "ms_short": ms_short,
            "ms_short_ratio": ms_short_ratio,
            "distance_threshold": distance_threshold,
            "channel_support": channel_support,
            "current_distances": current_distances,
            "association_profile": association_profile,
            "reliability": reliability,
            "memory_patch_ids": memory_patch_ids,
            "expert_supports": expert_supports,
        }

    def commit(self, prepared):
        step = prepared["step"]
        event = self.manager.advance(
            step=step,
            image_id=step,
            dino_patch_features=prepared["current"],
            ms_score=prepared["ms_score"],
            channel_support=prepared["channel_support"],
            expert_channel_supports=prepared["expert_supports"],
            provisional_channel_support=prepared["association_profile"][
                "recent_provisional_match_fraction"
            ],
            ms_short_score=prepared["ms_short"],
            ms_short_ratio=prepared["ms_short_ratio"],
            memory_patch_ids=prepared["memory_patch_ids"],
        )
        if prepared["current_distances"] is not None:
            finite = prepared["current_distances"][
                torch.isfinite(prepared["current_distances"])
            ]
            self.distance_history.extend(finite.tolist())
        soft_reliability = 0.5 + 0.5 * prepared["reliability"]
        self.memory.update(
            prepared["current"],
            image_id=step,
            grid_size=prepared["grid_size"],
            patch_reliabilities=soft_reliability,
        )
        self.historical_features.append(prepared["current"].detach().cpu())
        return {
            "step": step,
            "active_experts_before": prepared["active_experts_before"],
            "provisional_expert_before": prepared["provisional_expert_before"],
            "fuser_training_members_before": prepared[
                "fuser_training_members_before"
            ],
            "candidate_route": event["candidate_route"],
            "admission_route": event["admission_route"],
            "admitted_expert_id": event["admitted_expert_id"],
            "deleted_expert_ids": event["deleted_expert_ids"],
            "active_experts_after": self.manager.active_experts_before_step(),
            "active_expert_ids_after": event["active_expert_ids_after_step"],
            "channel_count_after": len(self.memory.channels),
            "mature_channel_count_after": len(self.memory.mature_channels()),
        }


def find_audit(experiment_root, sequence, seed, variant):
    sequence_name = "_to_".join(sequence)
    root = experiment_root / sequence_name / f"seed_{seed}" / variant
    matches = list(root.rglob(f"{sequence_name}_dynamic_audit_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{root} 下应有一个审计文件，实际为 {len(matches)} 个")
    return json.loads(matches[0].read_text(encoding="utf-8")), matches[0]


def online_score(prepared, history, device, image_size):
    step = prepared["step"]
    current = prepared["current"]
    active = prepared["active_experts_before"]
    provisional = prepared["provisional_expert_before"]
    expert_image_ids = [int(expert["image_id"]) for expert in active]
    if expert_image_ids:
        patch_score = dino_patch_score(
            current,
            [history[index]["features"] for index in expert_image_ids],
            device,
            topmin_min=0.02,
        )
        effective_mode = "dino_only"
        fallback_reason = None
    else:
        patch_score = dino_patch_score(
            current,
            [sample["features"] for sample in history],
            device,
            topmin_min=0.0,
        )
        effective_mode = "dino_fallback"
        fallback_reason = (
            "strict_history_dino" if step else "first_image_unavailable"
        )

    provisional_ids = []
    if provisional is not None:
        provisional_ids = [
            int(value)
            for value in provisional.get("image_ids", [provisional["image_id"]])
        ]
        provisional_score = dino_patch_score(
            current,
            [history[index]["features"] for index in provisional_ids],
            device,
            topmin_min=0.02,
        )
        weight = float(provisional["weight"])
        patch_score = (1.0 - weight) * patch_score + weight * provisional_score
        effective_mode = "dino_provisional"

    score_map = interpolate_patch_score(patch_score, image_size)
    audit = {
        "effective_mode": effective_mode,
        "active_expert_ids": [int(expert["expert_id"]) for expert in active],
        "expert_image_ids": expert_image_ids,
        "provisional_expert_image_id": (
            int(provisional["image_id"]) if provisional is not None else None
        ),
        "provisional_expert_image_ids": provisional_ids,
        "provisional_state": (
            provisional.get("state") if provisional is not None else None
        ),
        "fallback_reason": fallback_reason,
    }
    return score_map, audit


def compare_discrete(actual, expected, fields, step, kind):
    differences = []
    for field in fields:
        if actual.get(field) != expected.get(field):
            differences.append(
                {
                    "step": step,
                    "kind": kind,
                    "field": field,
                    "online": actual.get(field),
                    "batch": expected.get(field),
                }
            )
    return differences


def main():
    args = parse_args()
    sequence = args.sequence.split("->")
    if len(sequence) != 2:
        raise ValueError("sequence 必须写成 source->target")
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_dinov2(args.dinov2_repo, args.checkpoint, device)
    extractor = CrossCategoryFeatureExtractor(model, device)
    transform = image_transform(
        args.image_size,
        is_train=False,
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )
    sample_stream = []
    for category in sequence:
        sample_stream.extend(
            selected_stream(
                args.data_root,
                category,
                args.seed,
                args.segment_samples,
                transform,
                args.image_size,
            )
        )

    experiment_root = Path(args.experiment_root).resolve()
    summary_path = experiment_root / "cross_category_switch_summary.json"
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = []
    for variant in args.variants:
        audit, audit_path = find_audit(
            experiment_root, sequence, args.seed, variant
        )
        state = OnlineTimelineState(device, variant)
        history = []
        online_maps = []
        batch_maps = []
        differences = []
        print(f"验证 {variant}: 0/{len(sample_stream)}", end="\r", flush=True)
        for step, sample in enumerate(sample_stream):
            features, _ = extractor.extract(sample["image"])
            current = features.to(device)
            prepared = state.prepare(current, step)
            online_map, online_scoring = online_score(
                prepared, history, device, args.image_size
            )
            timeline_row = state.commit(prepared)
            online_maps.append(online_map.numpy())
            online_sample = {"features": features}
            history.append(online_sample)

            expected_timeline = audit["timeline"][step]
            expected_scoring = audit["scoring"][step]
            differences.extend(
                compare_discrete(
                    timeline_row,
                    expected_timeline,
                    DISCRETE_TIMELINE_FIELDS,
                    step,
                    "timeline",
                )
            )
            differences.extend(
                compare_discrete(
                    online_scoring,
                    expected_scoring,
                    DISCRETE_SCORING_FIELDS,
                    step,
                    "scoring",
                )
            )
            expected_map, _ = online_score(
                {
                    "step": step,
                    "current": current,
                    "active_experts_before": expected_timeline[
                        "active_experts_before"
                    ],
                    "provisional_expert_before": expected_timeline.get(
                        "provisional_expert_before"
                    ),
                },
                history[:-1],
                device,
                args.image_size,
            )
            batch_maps.append(expected_map.numpy())
            print(
                f"验证 {variant}: {step + 1}/{len(sample_stream)}",
                end="\r",
                flush=True,
            )
        print(" " * 100, end="\r", flush=True)
        online_maps = np.stack(online_maps)
        batch_maps = np.stack(batch_maps)
        map_error = float(np.max(np.abs(online_maps - batch_maps)))
        online_scores = online_maps.reshape(len(online_maps), -1).max(axis=1)
        batch_scores = batch_maps.reshape(len(batch_maps), -1).max(axis=1)
        score_error = float(np.max(np.abs(online_scores - batch_scores)))
        labels = np.asarray(
            [int(sample["is_anomaly"]) for sample in sample_stream]
        )
        online_auroc = float(roc_auc_score(labels, online_scores))
        batch_auroc = float(roc_auc_score(labels, batch_scores))
        matching_runs = [
            run
            for run in source_summary["runs"]
            if run["sequence"] == args.sequence
            and int(run["seed"]) == args.seed
            and run["variant"] == variant
        ]
        if len(matching_runs) != 1:
            raise ValueError(
                f"原汇总中应有一个匹配实验，实际为 {len(matching_runs)} 个"
            )
        source_raw_auroc = float(matching_runs[0]["raw_image_auroc"])
        auroc_error = abs(online_auroc - source_raw_auroc)
        equivalent = (
            not differences
            and map_error <= args.float_tolerance
            and score_error <= args.float_tolerance
            and auroc_error <= args.float_tolerance
        )
        results.append(
            {
                "variant": variant,
                "audit_path": str(audit_path),
                "step_count": len(sample_stream),
                "timeline_and_reference_difference_count": len(differences),
                "first_differences": differences[:20],
                "max_score_map_abs_error": map_error,
                "max_raw_image_score_abs_error": score_error,
                "online_raw_image_auroc": online_auroc,
                "batch_replay_raw_image_auroc": batch_auroc,
                "source_summary_raw_image_auroc": source_raw_auroc,
                "source_summary_raw_image_auroc_abs_error": auroc_error,
                "equivalent": equivalent,
            }
        )

    report = {
        "protocol": {
            "sequence": args.sequence,
            "seed": args.seed,
            "segment_samples": args.segment_samples,
            "rscin_enabled": False,
            "order": "extract_one_image -> score_with_step_start_state -> update_state",
            "float_tolerance": args.float_tolerance,
        },
        "results": results,
        "all_equivalent": all(result["equivalent"] for result in results),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cross_category_stream_equivalence.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"结果已写入: {output_path}")
    if not report["all_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
