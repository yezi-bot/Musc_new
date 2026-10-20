import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from datasets.gradual_drift import build_lab_feature_stream
from examples.ablate_gradual_drift import (
    build_lab_audit_profiles,
    build_lab_ewma_profiles,
    build_monotonicity_summary,
    fixed_initial_expert,
    lab_ewma_metrics,
    lab_metrics,
)


def read_csv(path):
    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
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


def run_directory(input_dir, run):
    return (
        Path(input_dir)
        / run["category"]
        / run["drift_type"]
        / run["drift_level"]
        / f"seed_{int(run['seed'])}"
        / run["variant"]
    )


def load_reference(input_dir, run):
    keys = (
        "fixed_reference_expert_id",
        "fixed_reference_image_id",
        "fixed_reference_admission_step",
    )
    if all(run.get(key) not in (None, "") for key in keys):
        return {
            "expert_id": int(run[keys[0]]),
            "image_id": int(run[keys[1]]),
            "admission_step": int(run[keys[2]]),
        }

    audit_files = list(
        run_directory(input_dir, run).rglob(
            f"{run['category']}_dynamic_audit_"
            f"seed{int(run['seed'])}_part0.json"
        )
    )
    if len(audit_files) != 1:
        raise RuntimeError(
            "expected exactly one dynamic audit file to recover "
            f"the initial expert, found {len(audit_files)}"
        )
    audit = json.loads(audit_files[0].read_text(encoding="utf-8"))
    reference = fixed_initial_expert(audit["timeline"])
    if reference is None:
        raise RuntimeError("the stream did not admit a formal expert")
    return reference


def audit_run(input_dir, run, args):
    steps = read_csv(
        run_directory(input_dir, run)
        / "gradual_drift_steps.csv"
    )
    records = [
        (None, row["anomaly_type"], row["image_path"], None)
        for row in steps
    ]
    schedule = [
        {
            "phase": row["drift_phase"],
            "brightness": float(row["brightness"]),
            "contrast": float(row["contrast"]),
            "saturation": float(row["saturation"]),
            "hue": float(row["hue"]),
        }
        for row in steps
    ]
    labels = np.asarray(
        [int(row["is_anomaly"]) for row in steps]
    )
    reference = load_reference(input_dir, run)
    lab_features = build_lab_feature_stream(records, schedule)
    lab_profiles = build_lab_audit_profiles(
        lab_features,
        reference=reference,
    )
    lab_ewma_profiles = build_lab_ewma_profiles(
        lab_features,
        reference=reference,
        alpha=args.lab_ewma_alpha,
        threshold_k=args.lab_ewma_threshold_k,
        baseline_window=args.lab_ewma_baseline_window,
        confirm_steps=args.lab_ewma_confirm_steps,
    )

    output_steps = []
    feature_names = (
        "lab_l_median",
        "lab_l_iqr",
        "lab_a_median",
        "lab_a_iqr",
        "lab_b_median",
        "lab_b_iqr",
    )
    for row, feature, profile, ewma_profile in zip(
        steps,
        lab_features,
        lab_profiles,
        lab_ewma_profiles,
    ):
        ewma_profile = ewma_profile or {}
        output = {
            "category": run["category"],
            "drift_type": run["drift_type"],
            "drift_level": run["drift_level"],
            "seed": int(run["seed"]),
            "variant": run["variant"],
            **row,
        }
        output.update(
            {
                name: float(value)
                for name, value in zip(feature_names, feature)
            }
        )
        output.update(
            {
                "lab_reference_expert_id": profile[
                    "reference_expert_id"
                ],
                "lab_reference_image_id": profile[
                    "reference_image_id"
                ],
                "lab_l_distance": profile["l_distance"],
                "lab_ab_distance": profile["ab_distance"],
                "lab_total_distance": profile["total_distance"],
                "lab_ewma_baseline_start": ewma_profile.get(
                    "baseline_start"
                ),
                "lab_ewma_baseline_end": ewma_profile.get(
                    "baseline_end"
                ),
                "lab_ewma_baseline_count": ewma_profile.get(
                    "baseline_count"
                ),
                "lab_l_baseline_median": ewma_profile.get(
                    "l_baseline_median"
                ),
                "lab_l_baseline_mad": ewma_profile.get("l_baseline_mad"),
                "lab_l_threshold": ewma_profile.get("l_threshold"),
                "lab_ab_baseline_median": ewma_profile.get(
                    "ab_baseline_median"
                ),
                "lab_ab_baseline_mad": ewma_profile.get(
                    "ab_baseline_mad"
                ),
                "lab_ab_threshold": ewma_profile.get("ab_threshold"),
                "lab_l_ewma": ewma_profile.get("l_ewma"),
                "lab_ab_ewma": ewma_profile.get("ab_ewma"),
                "lab_l_hit": ewma_profile.get("l_hit"),
                "lab_ab_hit": ewma_profile.get("ab_hit"),
                "lab_l_streak": ewma_profile.get("l_streak"),
                "lab_ab_streak": ewma_profile.get("ab_streak"),
                "lab_l_confirmed": ewma_profile.get("l_confirmed"),
                "lab_ab_confirmed": ewma_profile.get("ab_confirmed"),
                "lab_drift_confirmed": ewma_profile.get(
                    "drift_confirmed"
                ),
            }
        )
        output_steps.append(output)

    output_run = dict(run)
    output_run.update(
        {
            "lab_ewma_alpha": args.lab_ewma_alpha,
            "lab_ewma_threshold_k": args.lab_ewma_threshold_k,
            "lab_ewma_baseline_window": args.lab_ewma_baseline_window,
            "lab_ewma_confirm_steps": args.lab_ewma_confirm_steps,
        }
    )
    output_run.update(lab_metrics(lab_profiles, schedule, labels))
    output_run.update(lab_ewma_metrics(lab_ewma_profiles, schedule))
    return output_run, output_steps


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Add fixed-initial-expert Lab sensitivity audit "
            "to an existing gradual-drift result."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lab-ewma-alpha", type=float, default=0.3)
    parser.add_argument("--lab-ewma-threshold-k", type=float, default=3.0)
    parser.add_argument("--lab-ewma-baseline-window", type=int, default=8)
    parser.add_argument("--lab-ewma-confirm-steps", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 < args.lab_ewma_alpha <= 1.0:
        raise ValueError("lab-ewma-alpha must be within (0, 1]")
    if args.lab_ewma_threshold_k < 0.0:
        raise ValueError("lab-ewma-threshold-k must be non-negative")
    if args.lab_ewma_baseline_window < 2:
        raise ValueError("lab-ewma-baseline-window must be at least 2")
    if args.lab_ewma_confirm_steps < 1:
        raise ValueError("lab-ewma-confirm-steps must be positive")

    input_dir = Path(args.input_dir)
    summary = json.loads(
        (input_dir / "gradual_drift_summary.json").read_text(
            encoding="utf-8"
        )
    )
    audited_runs = []
    audited_steps = []
    for run in summary["runs"]:
        audited_run, run_steps = audit_run(input_dir, run, args)
        audited_runs.append(audited_run)
        audited_steps.extend(run_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "lab_sensitivity_runs.csv", audited_runs)
    write_csv(output_dir / "lab_sensitivity_steps.csv", audited_steps)

    metrics = {
        "lab_l_distance_monotonicity": (
            "plateau_lab_l_distance_median"
        ),
        "lab_ab_distance_monotonicity": (
            "plateau_lab_ab_distance_median"
        ),
        "lab_total_distance_monotonicity": (
            "plateau_lab_total_distance_median"
        ),
        "lab_total_clean_distance_monotonicity": (
            "plateau_lab_total_distance_clean_median"
        ),
    }
    report = {"runs": audited_runs}
    for name, metric in metrics.items():
        report[name] = build_monotonicity_summary(
            audited_runs,
            metric=metric,
            require_robust_shift_audit=False,
        )
    (output_dir / "lab_sensitivity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
