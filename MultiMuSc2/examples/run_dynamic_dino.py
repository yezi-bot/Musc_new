import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mvtec import DatasetSplit, MVTecDataset
from models.modules._DYNAMIC_DINO_STREAM import (
    DynamicDinoOnlineState,
    feature_key,
)
from models.modules._LNAMD import LNAMD
from utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strictly online Dynamic DINO-only evaluation."
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
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def load_dinov2(repo_path, checkpoint_path, device):
    model = torch.hub.load(
        str(Path(repo_path).resolve()),
        "dinov2_vitl14",
        source="local",
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "teacher"):
            if key in checkpoint and isinstance(
                checkpoint[key], dict
            ):
                checkpoint = checkpoint[key]
                break

    cleaned = {}
    for key, value in checkpoint.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    model.requires_grad_(False)
    model.to(device)
    return model


class DinoFeatureExtractor:
    def __init__(
        self,
        model,
        device,
        feature_layers,
        r_list,
    ):
        self.model = model
        self.device = torch.device(device)
        self.feature_layers = [
            int(layer) for layer in feature_layers
        ]
        self.r_list = [int(r) for r in r_list]

        self.aggregators = {
            r: LNAMD(
                device=self.device,
                r=r,
                feature_dim=1024,
                feature_layer=[
                    layer + 1
                    for layer in self.feature_layers
                ],
            )
            for r in self.r_list
        }

    def extract(self, image):
        image = image.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type == "cuda",
            ):
                patch_tokens = (
                    self.model.get_intermediate_layers(
                        image,
                        n=self.feature_layers,
                        return_class_token=False,
                    )
                )

                patch_tokens = [
                    torch.cat(
                        [
                            torch.zeros_like(tokens[:, :1]),
                            tokens,
                        ],
                        dim=1,
                    )
                    for tokens in patch_tokens
                ]

                features = {}
                for r, aggregator in self.aggregators.items():
                    aggregated = aggregator._embed(
                        patch_tokens
                    ).float()
                    aggregated = F.normalize(
                        aggregated,
                        dim=-1,
                    )

                    for layer_index, layer in enumerate(
                        self.feature_layers
                    ):
                        features[feature_key(r, layer)] = (
                            aggregated[
                                0,
                                :,
                                layer_index,
                                :,
                            ]
                            .detach()
                            .cpu()
                        )
        return features


def select_dataset(dataset, seed, max_samples):
    indices = list(range(len(dataset)))

    if max_samples is not None:
        if max_samples < 2:
            raise ValueError(
                "--max-samples must be at least 2"
            )
        if max_samples < len(indices):
            indices = (
                torch.linspace(
                    0,
                    len(indices) - 1,
                    steps=max_samples,
                )
                .round()
                .long()
                .unique(sorted=True)
                .tolist()
            )

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(
        len(indices),
        generator=generator,
    ).tolist()
    shuffled = [indices[index] for index in order]
    return torch.utils.data.Subset(dataset, shuffled)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open(
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


def run_seed(
    cfg,
    args,
    seed,
    model,
    device,
    output_dir,
):
    seed_everything(seed)

    dataset = MVTecDataset(
        source=args.data_root,
        classname=cfg["datasets"]["class_name"],
        resize=int(cfg["datasets"]["img_resize"]),
        imagesize=int(cfg["datasets"]["img_resize"]),
        split=DatasetSplit.TEST,
    )
    dataset = select_dataset(
        dataset,
        seed,
        args.max_samples,
    )

    extractor = DinoFeatureExtractor(
        model=model,
        device=device,
        feature_layers=cfg["models"]["feature_layers"],
        r_list=cfg["models"]["r_list"],
    )
    online_state = DynamicDinoOnlineState(
        device=device,
        image_size=int(cfg["datasets"]["img_resize"]),
        feature_layers=cfg["models"]["feature_layers"],
        r_list=cfg["models"]["r_list"],
        committee_config=cfg["models"][
            "dynamic_committee"
        ],
        scoring_config=cfg["models"]["scoring"],
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    available_labels = []
    available_scores = []
    available_masks = []
    available_maps = []
    audit = []

    for step in range(len(dataset)):
        sample = dataset[step]
        current_features = extractor.extract(
            sample["image"].unsqueeze(0)
        )

        anomaly_map, record = (
            online_state.process_features(
                current_features
            )
        )

        record["image_path"] = os.path.relpath(
            sample["image_path"],
            args.data_root,
        )
        record["anomaly_type"] = Path( sample["image_path"]).parent.name
        audit.append(record)

        if anomaly_map is None:
            continue

        available_labels.append(
            int(sample["is_anomaly"])
        )
        available_scores.append(
            float(record["image_score"])
        )
        available_masks.append((sample["mask"] > 0.5).numpy().astype(np.int32))
        available_maps.append(
            anomaly_map.numpy()
        )

    runtime_seconds = time.perf_counter() - started

    gt_sp = np.asarray(
        available_labels,
        dtype=np.int32,
    )
    pr_sp = np.asarray(
        available_scores,
        dtype=np.float64,
    )
    gt_px = np.stack(available_masks)
    pr_px = np.stack(available_maps)
# 计算image/pixel指标
    image_metric, pixel_metric = compute_metrics(
        gt_sp=gt_sp,
        pr_sp=pr_sp,
        gt_px=gt_px,
        pr_px=pr_px,
    )

    result = {
        "seed": seed,
        "sample_count": len(dataset),
        "available_count": len(available_scores),
        "unavailable_count":
            len(dataset) - len(available_scores),
        "coverage":
            len(available_scores) / len(dataset),
        "image_auroc": float(image_metric[0]),
        "image_f1": float(image_metric[1]),
        "image_ap": float(image_metric[2]),
        "pixel_auroc": float(pixel_metric[0]),
        "pixel_f1": float(pixel_metric[1]),
        "pixel_ap": float(pixel_metric[2]),
        "aupro": float(pixel_metric[3]),
        "runtime_seconds": runtime_seconds,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(device)
            / 1024
            / 1024
            if device.type == "cuda"
            else 0.0
        ),
    }

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "timeline.json").write_text(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (seed_dir / "metrics.json").write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result

# 入口
def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        cfg = yaml.safe_load(handle)
# dino-only
    if (
        cfg["models"]["dynamic_fusion"]["mode"]
        != "dino_only"
    ):
        raise ValueError(
            "run_dynamic_dino.py only accepts dino_only mode"
        )
        # 禁止RsCIN
    if cfg["testing"].get("use_rscin", False):
        raise ValueError(
            "strict online mode does not allow RsCIN"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )
    model = load_dinov2(
        args.dinov2_repo,
        args.checkpoint,
        device,
    )

    results = []
    for seed in args.seeds:
        results.append(
            run_seed(
                cfg,
                args,
                seed,
                model,
                device,
                output_dir,
            )
        )

    write_csv(
        output_dir / "dynamic_dino_runs.csv",
        results,
    )

    summary = {}
    for key in results[0]:
        if key == "seed":
            continue
        values = [
            row[key]
            for row in results
            if isinstance(row[key], (int, float))
        ]
        summary[f"{key}_mean"] = (
            sum(values) / len(values)
        )

    metadata = {
        "git_commit": git_commit(),
        "checkpoint": str(
            Path(args.checkpoint).resolve()
        ),
        "checkpoint_sha256": sha256_file(
            args.checkpoint
        ),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "config": cfg,
        "seeds": args.seeds,
        "max_samples": args.max_samples,
        "runs": results,
        "summary": summary,
    }
    (output_dir / "dynamic_dino_summary.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()