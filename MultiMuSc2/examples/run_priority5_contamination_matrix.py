import argparse
import json
import subprocess
import sys
from pathlib import Path


VARIANTS = [
    "priority5_task_aware_virtual_swap",
    "priority5_task_aware_soft_reliability",
    "priority5_task_aware_cross_patch_reliability",
]


def completed(output_dir, drift_types, drift_levels, seeds):
    summary_path = output_dir / "gradual_drift_summary.json"
    if not summary_path.exists():
        return False
    try:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))["runs"]
    except (KeyError, json.JSONDecodeError, OSError):
        return False
    expected = {
        (drift_type, drift_level, int(seed), variant)
        for drift_type in drift_types
        for drift_level in drift_levels
        for seed in seeds
        for variant in VARIANTS
    }
    actual = {
        (
            row.get("drift_type"),
            row.get("drift_level"),
            int(row.get("seed", -1)),
            row.get("variant"),
        )
        for row in rows
    }
    return expected <= actual


def main():
    parser = argparse.ArgumentParser(
        description="运行第五阶段真实 patch 污染 A/B/C 配对矩阵。"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["bottle", "cable", "capsule", "metal_nut"],
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument(
        "--drift-types", nargs="+", default=["brightness", "color"]
    )
    parser.add_argument("--drift-levels", nargs="+", default=["strong"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--stream-samples", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).with_name("ablate_gradual_drift.py")
    output_root = Path(args.output_root).resolve()
    for category in args.categories:
        for ratio in args.ratios:
            ratio_tag = int(round(100 * ratio))
            output_dir = output_root / f"{category}-patch-{ratio_tag}"
            if args.resume and completed(
                output_dir,
                args.drift_types,
                args.drift_levels,
                args.seeds,
            ):
                print(f"Skipping complete condition: {output_dir}")
                continue
            command = [
                sys.executable,
                str(script),
                "--data-root",
                str(Path(args.data_root).resolve()),
                "--output-dir",
                str(output_dir),
                "--category",
                category,
                "--drift-types",
                *args.drift_types,
                "--drift-levels",
                *args.drift_levels,
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--stream-samples",
                str(args.stream_samples),
                "--anomaly-ratio",
                str(ratio),
                "--controlled-contamination",
                "--contamination-mode",
                "patch",
                "--contamination-patch-ratio",
                str(ratio),
                "--variants",
                *VARIANTS,
            ]
            print(f"Running condition: {category}, ratio={ratio}")
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
