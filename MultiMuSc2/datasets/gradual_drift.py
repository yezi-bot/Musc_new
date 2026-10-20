import random

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2lab
from torchvision.transforms import functional as TF


PHASES = (
    "cold_start",
    "ramp_up",
    "plateau",
    "ramp_down",
    "recovery",
)


def apply_photometric_drift(image, drift):
    image = TF.adjust_brightness(
        image,
        float(drift["brightness"]),
    )
    image = TF.adjust_contrast(
        image,
        float(drift["contrast"]),
    )
    image = TF.adjust_saturation(
        image,
        float(drift["saturation"]),
    )
    return TF.adjust_hue(
        image,
        float(drift["hue"]),
    )


def lab_robust_feature(image):
    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ) / 255.0
    lab = rgb2lab(rgb)
    feature = []
    for channel in range(3):
        values = lab[..., channel].reshape(-1)
        q25, median, q75 = np.quantile(
            values,
            [0.25, 0.5, 0.75],
        )
        feature.extend(
            [
                float(median),
                float(q75 - q25),
            ]
        )
    return np.asarray(feature, dtype=np.float64)


def build_lab_feature_stream(records, drift_schedule):
    if len(records) != len(drift_schedule):
        raise ValueError(
            "records and drift_schedule must have the same length"
        )

    features = []
    for record, drift in zip(records, drift_schedule):
        image_path = record[2]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        image = apply_photometric_drift(image, drift)
        features.append(lab_robust_feature(image))

    return np.stack(features, axis=0)


class GradualDriftDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records,
        transform_img,
        transform_mask,
        drift_schedule,
    ):
        if len(records) != len(drift_schedule):
            raise ValueError(
                "records and drift_schedule must have the same length"
            )

        self.records = list(records)
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.drift_schedule = list(drift_schedule)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        classname, anomaly, image_path, mask_path = self.records[index]
        drift = self.drift_schedule[index]

        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        image = apply_photometric_drift(image, drift)
        lab_feature = torch.from_numpy(
            lab_robust_feature(image)
        ).to(torch.float32)

        image = self.transform_img(image)

        if mask_path is not None:
            with Image.open(mask_path) as handle:
                mask = self.transform_mask(
                    handle.convert("L")
                )
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(
                [1, image.shape[-2], image.shape[-1]],
                dtype=torch.float32,
            )

        return {
            "image": image,
            "mask": mask,
            "is_anomaly": int(anomaly != "good"),
            "image_path": str(image_path),
            "stream_step": int(index),
            "drift_phase": drift["phase"],
            "drift_severity": float(drift["severity"]),
            "lab_feature": lab_feature,
        }


def drift_state(step, total_steps):
    if total_steps < 10:
        raise ValueError(
            "total_steps must be at least 10"
        )

    if total_steps % len(PHASES) != 0:
        raise ValueError(
            "total_steps must be divisible by 5"
        )

    if step < 0 or step >= total_steps:
        raise ValueError(
            "step must be within the stream"
        )

    phase_size = total_steps // len(PHASES)
    phase_index = min(
        step // phase_size,
        len(PHASES) - 1,
    )
    phase = PHASES[phase_index]

    local_step = (
        step - phase_index * phase_size
    )

    if phase == "cold_start":
        severity = 0.0
    elif phase == "ramp_up":
        severity = (
            local_step / max(phase_size - 1, 1)
        )
    elif phase == "plateau":
        severity = 1.0
    elif phase == "ramp_down":
        severity = 1.0 - (
            local_step / max(phase_size - 1, 1)
        )
    else:
        severity = 0.0

    return float(severity), phase


def build_drift_schedule(
    drift_type,
    total_steps,
    brightness_delta=0.15,
    contrast_delta=0.05,
    saturation_delta=0.15,
    hue_delta=0.02,
):
    if drift_type not in {
        "brightness",
        "color",
    }:
        raise ValueError(
            f"unsupported drift type: {drift_type}"
        )

    schedule = []

    for step in range(total_steps):
        severity, phase = drift_state(
            step,
            total_steps,
        )

        brightness = 1.0
        contrast = 1.0
        saturation = 1.0
        hue = 0.0

        if drift_type == "brightness":
            brightness += (
                brightness_delta * severity
            )
            contrast += (
                contrast_delta * severity
            )
        elif drift_type == "color":
            saturation += (
                saturation_delta * severity
            )
            hue += hue_delta * severity

        if brightness <= 0.0:
            raise ValueError(
                "brightness factor must be positive"
            )

        if contrast <= 0.0:
            raise ValueError(
                "contrast factor must be positive"
            )

        if saturation < 0.0:
            raise ValueError(
                "saturation factor must be non-negative"
            )

        if not -0.5 <= hue <= 0.5:
            raise ValueError(
                "hue factor must be within [-0.5, 0.5]"
            )

        schedule.append(
            {
                "step": int(step),
                "phase": phase,
                "severity": float(severity),
                "brightness": float(brightness),
                "contrast": float(contrast),
                "saturation": float(saturation),
                "hue": float(hue),
            }
        )

    return schedule


def build_balanced_stream_records(
    records,
    total_steps,
    anomaly_ratio,
    seed,
):
    if not 0.0 < anomaly_ratio < 1.0:
        raise ValueError(
            "anomaly_ratio must be within (0, 1)"
        )

    if total_steps < 10:
        raise ValueError(
            "total_steps must be at least 10"
        )

    if total_steps % len(PHASES) != 0:
        raise ValueError(
            "total_steps must be divisible by 5"
        )

    normal_records = [
        tuple(record)
        for record in records
        if record[1] == "good"
    ]

    anomaly_records = [
        tuple(record)
        for record in records
        if record[1] != "good"
    ]

    generator = random.Random(seed)
    generator.shuffle(normal_records)
    generator.shuffle(anomaly_records)

    phase_size = (
        total_steps // len(PHASES)
    )

    anomaly_per_phase = round(
        phase_size * anomaly_ratio
    )
    anomaly_per_phase = max(
        1,
        min(
            phase_size - 1,
            anomaly_per_phase,
        ),
    )
    normal_per_phase = (
        phase_size - anomaly_per_phase
    )

    required_normal = (
        normal_per_phase * len(PHASES)
    )
    required_anomaly = (
        anomaly_per_phase * len(PHASES)
    )

    if len(normal_records) < required_normal:
        raise ValueError(
            f"not enough normal images: "
            f"need {required_normal}, "
            f"found {len(normal_records)}; "
            f"reduce stream samples or "
            f"increase anomaly ratio"
        )

    if len(anomaly_records) < required_anomaly:
        raise ValueError(
            f"not enough anomaly images: "
            f"need {required_anomaly}, "
            f"found {len(anomaly_records)}"
        )

    stream_records = []
    normal_offset = 0
    anomaly_offset = 0

    for phase in PHASES:
        phase_records = []

        phase_records.extend(
            normal_records[
                normal_offset:
                normal_offset + normal_per_phase
            ]
        )

        phase_records.extend(
            anomaly_records[
                anomaly_offset:
                anomaly_offset + anomaly_per_phase
            ]
        )

        normal_offset += normal_per_phase
        anomaly_offset += anomaly_per_phase

        generator.shuffle(phase_records)
        stream_records.extend(phase_records)

    return stream_records
