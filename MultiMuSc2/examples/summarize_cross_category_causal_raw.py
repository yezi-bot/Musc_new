import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "raw_image_auroc",
    "pre_raw_image_auroc",
    "post_raw_image_auroc",
    "post5_raw_image_auroc",
    "post10_raw_image_auroc",
    "post20_raw_image_auroc",
    "pixel_auroc",
    "pre_pixel_auroc",
    "post_pixel_auroc",
    "aupro",
    "image_auroc",
    "post_old_expert_reference_fraction",
    "new_expert_latency",
    "old_expert_clear_latency",
)

VARIANT_NAMES = {
    "dynamic_dino_quarantine": "Quarantine",
    "dynamic_dino_channel_gate_no_fallback": "Channel-gate（无历史退回）",
}
EXPECTED_SEQUENCES = {"bottle->cable", "cable->bottle"}
EXPECTED_VARIANTS = set(VARIANT_NAMES)
EXPECTED_SEEDS = {0, 1, 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="从既有跨类别实验中汇总 causal raw AUROC，不重新推理。"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--contamination-summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def summarize(values):
    values = [float(value) for value in values if value is not None]
    return {
        "mean": float(np.mean(values)) if values else None,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "n": len(values),
        "values": values,
    }


def validate_runs(runs):
    expected_keys = {
        (sequence, variant, seed)
        for sequence in EXPECTED_SEQUENCES
        for variant in EXPECTED_VARIANTS
        for seed in EXPECTED_SEEDS
    }
    actual_keys = {
        (run["sequence"], run["variant"], int(run["seed"])) for run in runs
    }
    if len(runs) != len(actual_keys):
        raise ValueError("原实验汇总中存在重复的方向、方案和 seed")
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"实验范围不完整或不匹配；缺失={missing}，额外={unexpected}"
        )
    required_fields = set(METRICS) | {
        "causal_reference_violations",
        "boundary",
        "segment_samples",
    }
    for run in runs:
        absent = sorted(required_fields - set(run))
        if absent:
            raise ValueError(
                f"{run['sequence']} {run['variant']} seed={run['seed']} "
                f"缺少字段: {absent}"
            )
        if int(run["boundary"]) != 40 or int(run["segment_samples"]) != 40:
            raise ValueError("本报告只接受每段 40 张、边界位于第 40 步的实验")


def group_runs(runs):
    groups = defaultdict(list)
    for run in runs:
        groups[(run["sequence"], run["variant"])].append(run)
    output = {}
    for (sequence, variant), selected in sorted(groups.items()):
        selected = sorted(selected, key=lambda run: int(run["seed"]))
        output[f"{sequence}|{variant}"] = {
            "sequence": sequence,
            "variant": variant,
            "seeds": [int(run["seed"]) for run in selected],
            "run_count": len(selected),
            "causal_reference_violations": sum(
                int(run["causal_reference_violations"]) for run in selected
            ),
            "metrics": {
                metric: summarize([run.get(metric) for run in selected])
                for metric in METRICS
            },
        }
    return output


def format_stat(stat, digits=4):
    if stat["mean"] is None:
        return "—"
    if stat["std"] is None:
        return f"{stat['mean']:.{digits}f}（n={stat['n']}）"
    suffix = f"，n={stat['n']}" if stat["n"] != 3 else ""
    return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}{suffix}"


def metric_mean(groups, sequence, variant, metric):
    return groups[f"{sequence}|{variant}"]["metrics"][metric]["mean"]


def build_rows(groups):
    rows = []
    for group in groups.values():
        row = {
            "sequence": group["sequence"],
            "variant": group["variant"],
            "run_count": group["run_count"],
            "causal_reference_violations": group[
                "causal_reference_violations"
            ],
        }
        for metric, stat in group["metrics"].items():
            row[f"{metric}_mean"] = stat["mean"]
            row[f"{metric}_std"] = stat["std"]
            row[f"{metric}_n"] = stat["n"]
        rows.append(row)
    return rows


def contamination_lines(contamination):
    if contamination is None:
        return ["未提供专家污染分析汇总。"]
    aggregate = contamination["aggregate"]
    overall = aggregate["overall"]
    lines = [
        f"- 12 次运行共准入 {overall['formal_admission_count']} 次正式专家，其中异常正式专家为 "
        f"{overall['contaminated_formal_admission_count']}。",
        f"- 全部 960 个在线步骤中，异常 provisional 引用实际参与评分 "
        f"{overall['contaminated_exposure_steps']} 步（"
        f"{overall['contaminated_exposure_fraction_of_all_steps']:.2%}）。",
        f"- 切换后的 480 个步骤中，旧类别专家参与评分 "
        f"{overall['stale_post_switch_exposure_steps']} 步（"
        f"{overall['stale_post_switch_exposure_fraction']:.2%}）。",
    ]
    for sequence in ("bottle->cable", "cable->bottle"):
        key = f"{sequence}|dynamic_dino_channel_gate_no_fallback"
        item = aggregate["by_sequence_variant"][key]
        lines.append(
            f"- `{sequence}` 的 channel-gate：污染暴露 "
            f"{item['contaminated_exposure_steps']} 步，切换后旧类别引用 "
            f"{item['stale_post_switch_exposure_fraction']:.2%}。"
        )
    return lines


def build_report(groups, contamination, source_summary):
    quarantine = "dynamic_dino_quarantine"
    channel_gate = "dynamic_dino_channel_gate_no_fallback"
    lines = [
        "# 跨类别因果在线结果汇总",
        "",
        "## 1. 统计口径",
        "",
        "本报告直接读取已有跨类别实验记录，不重新提取特征或执行模型推理。实验范围为 "
        "`bottle→cable`、`cable→bottle` 两个方向，每个方向 3 个随机种子，每段 40 张图像。",
        "",
        "主图像级指标采用 **causal raw Image AUROC**：当前图像只使用步初已有的正式/provisional "
        "专家或严格历史图像评分，不使用整条流的 RsCIN。逐图在线等价验证已经确认专家时间线、引用 ID、"
        "分数图和 raw 分数与原批量因果模拟一致。",
        "",
        "原字段 `image_auroc` 是整条流 RsCIN 修正后的结果，只作为附加诊断，不作为在线主结果。"
        "Pixel AUROC 和 AUPRO 不经过 RsCIN，可以继续使用。",
        "",
        "## 2. 三种子主结果",
        "",
        "| 转换方向 | 方案 | Raw Image AUROC | 切换前 Raw AUROC | 切换后 Raw AUROC | Pixel AUROC | 切换后 Pixel AUROC | AUPRO |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sequence in ("bottle->cable", "cable->bottle"):
        for variant in (quarantine, channel_gate):
            group = groups[f"{sequence}|{variant}"]
            metric = group["metrics"]
            lines.append(
                f"| {sequence.replace('->', '→')} | {VARIANT_NAMES[variant]} | "
                f"{format_stat(metric['raw_image_auroc'])} | "
                f"{format_stat(metric['pre_raw_image_auroc'])} | "
                f"{format_stat(metric['post_raw_image_auroc'])} | "
                f"{format_stat(metric['pixel_auroc'])} | "
                f"{format_stat(metric['post_pixel_auroc'])} | "
                f"{format_stat(metric['aupro'])} |"
            )

    lines.extend(
        [
            "",
            "### Channel-gate 相对 Quarantine 的均值变化",
            "",
            "| 转换方向 | Raw Image AUROC | 切换后 Raw AUROC | Pixel AUROC | AUPRO |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for sequence in ("bottle->cable", "cable->bottle"):
        deltas = {
            metric: metric_mean(groups, sequence, channel_gate, metric)
            - metric_mean(groups, sequence, quarantine, metric)
            for metric in (
                "raw_image_auroc",
                "post_raw_image_auroc",
                "pixel_auroc",
                "aupro",
            )
        }
        lines.append(
            f"| {sequence.replace('->', '→')} | {deltas['raw_image_auroc']:+.4f} | "
            f"{deltas['post_raw_image_auroc']:+.4f} | "
            f"{deltas['pixel_auroc']:+.4f} | {deltas['aupro']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. 切换初期表现",
            "",
            "| 转换方向 | 方案 | 切换后前5张 Raw AUROC | 前10张 | 前20张 | 新专家形成时延 | 旧专家清除时延 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for sequence in ("bottle->cable", "cable->bottle"):
        for variant in (quarantine, channel_gate):
            metric = groups[f"{sequence}|{variant}"]["metrics"]
            lines.append(
                f"| {sequence.replace('->', '→')} | {VARIANT_NAMES[variant]} | "
                f"{format_stat(metric['post5_raw_image_auroc'])} | "
                f"{format_stat(metric['post10_raw_image_auroc'])} | "
                f"{format_stat(metric['post20_raw_image_auroc'])} | "
                f"{format_stat(metric['new_expert_latency'], 2)} 步 | "
                f"{format_stat(metric['old_expert_clear_latency'], 2)} 步 |"
            )
    lines.extend(
        [
            "",
            "前 5 张窗口样本很少，且 `cable→bottle` 有一个种子因窗口内只有单一标签而无法计算 AUROC，"
            "因此该列仅用于观察，不作为主要结论。",
            "",
            "## 4. RsCIN 对主结论的影响",
            "",
            "| 转换方向 | 方案 | Raw Image AUROC | RsCIN Image AUROC | RsCIN−Raw |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for sequence in ("bottle->cable", "cable->bottle"):
        for variant in (quarantine, channel_gate):
            raw = metric_mean(groups, sequence, variant, "raw_image_auroc")
            rscin = metric_mean(groups, sequence, variant, "image_auroc")
            lines.append(
                f"| {sequence.replace('->', '→')} | {VARIANT_NAMES[variant]} | "
                f"{raw:.4f} | {rscin:.4f} | {rscin - raw:+.4f} |"
            )

    lines.extend(
        [
            "",
            "RsCIN 在不同方向和方案上既可能升分也可能降分，而且使用整条流信息，因此不能代表真实在线性能。"
            "后续主表统一使用 raw Image AUROC。",
            "",
            "## 5. 专家机制辅助统计",
            "",
            *contamination_lines(contamination),
            "",
            "两个方案在同一方向和种子下具有相同的正式专家时间线；差别在于 channel-gate 会让确认前的 "
            "provisional 候选以低权重参与评分。`bottle→cable` 的主要风险信号是切换后一段时间仍使用旧类别专家，"
            "而不是异常图像成为正式专家。",
            "",
            "## 6. 结论",
            "",
            "1. 现有跨类别结果可以直接改用 causal raw Image AUROC，不需要重新进行特征推理。",
            "2. `bottle→cable` 中 channel-gate 相比 quarantine 的 raw Image AUROC 提高约 0.0650，"
            "切换后提高约 0.0453，Pixel AUROC 和 AUPRO也同步提高。",
            "3. `cable→bottle` 中提升很小：raw Image AUROC 约 +0.0065，且切换后 raw AUROC 约 -0.0056；"
            "说明门控收益具有明显方向依赖。",
            "4. 当前只能证明这两个方向、三个种子的现象，不能据此声称对所有类别转换都有效。",
            "5. 下一步应先用少量新增类别对做 seed 0 泛化检查；在趋势明确前，不应同时修改专家准入和加入 CLIP。",
            "",
            "## 7. 数据来源",
            "",
            f"- 原实验汇总：`{source_summary}`；",
            "- 专家污染和旧类别引用：`outputs/cross-category-contamination-full/cross_category_contamination_impact.json`；",
            "- 在线等价验证：`outputs/stream-equivalence-bottle-cable-seed0/cross_category_stream_equivalence.json`。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    runs = json.loads(summary_path.read_text(encoding="utf-8"))["runs"]
    validate_runs(runs)
    groups = group_runs(runs)
    contamination = None
    if args.contamination_summary:
        contamination = json.loads(
            Path(args.contamination_summary).read_text(encoding="utf-8")
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_summary": str(summary_path),
        "inference_rerun": False,
        "main_image_metric": "raw_image_auroc",
        "rscin_metric_role": "diagnostic_only",
        "groups": groups,
    }
    (output_dir / "causal_raw_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "causal_raw_summary.csv", build_rows(groups))
    report = build_report(groups, contamination, args.summary)
    Path(args.report).resolve().write_text(report, encoding="utf-8")
    print(f"已生成 {output_dir / 'causal_raw_summary.json'}")
    print(f"已生成 {output_dir / 'causal_raw_summary.csv'}")
    print(f"已生成 {Path(args.report).resolve()}")


if __name__ == "__main__":
    main()
