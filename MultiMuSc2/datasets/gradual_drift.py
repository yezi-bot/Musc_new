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


MULTISTAGE_PHOTOMETRIC_STATES = (
    {
        "state_id": "a",
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "hue": 0.0,
    },
    {
        "state_id": "b",
        "brightness": 1.25,
        "contrast": 1.08,
        "saturation": 1.0,
        "hue": 0.0,
    },
    {
        "state_id": "c",
        "brightness": 1.15,
        "contrast": 1.05,
        "saturation": 1.30,
        "hue": 0.04,
    },
    {
        "state_id": "d",
        "brightness": 0.80,
        "contrast": 0.95,
        "saturation": 0.85,
        "hue": -0.04,
    },
    {
        "state_id": "a_recovery",
        "brightness": 1.0,
        "contrast": 1.0,
        "saturation": 1.0,
        "hue": 0.0,
    },
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
            "drift_state_id": drift.get("state_id"),
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


def _photometric_severity(state):
    return float(
        min(
            1.0,
            max(
                abs(float(state["brightness"]) - 1.0) / 0.25,
                abs(float(state["contrast"]) - 1.0) / 0.08,
                abs(float(state["saturation"]) - 1.0) / 0.30,
                abs(float(state["hue"])) / 0.04,
            ),
        )
    )


def build_multistage_drift_schedule(
    initial_plateau_steps=4,
    transition_steps=2,
    plateau_steps=4,
):
    if initial_plateau_steps < 4:
        raise ValueError("initial_plateau_steps must be at least 4")
    if transition_steps < 1:
        raise ValueError("transition_steps must be positive")
    if plateau_steps < 3:
        raise ValueError("plateau_steps must be at least 3")

    schedule = []

    def append_state(state, phase, segment_kind):
        schedule.append(
            {
                "step": len(schedule),
                "phase": phase,
                "segment_id": phase,
                "segment_kind": segment_kind,
                "state_id": state["state_id"],
                "severity": _photometric_severity(state),
                "brightness": float(state["brightness"]),
                "contrast": float(state["contrast"]),
                "saturation": float(state["saturation"]),
                "hue": float(state["hue"]),
            }
        )

    initial = MULTISTAGE_PHOTOMETRIC_STATES[0]
    for _ in range(initial_plateau_steps):
        append_state(initial, "state_a_initial", "plateau")

    previous = initial
    for target in MULTISTAGE_PHOTOMETRIC_STATES[1:]:
        transition_id = (
            f"transition_{previous['state_id']}_to_{target['state_id']}"
        )
        for offset in range(1, transition_steps + 1):
            fraction = offset / float(transition_steps + 1)
            interpolated = {
                "state_id": transition_id,
                **{
                    key: (
                        float(previous[key])
                        + fraction
                        * (float(target[key]) - float(previous[key]))
                    )
                    for key in (
                        "brightness",
                        "contrast",
                        "saturation",
                        "hue",
                    )
                },
            }
            append_state(interpolated, transition_id, "transition")

        plateau_id = f"state_{target['state_id']}"
        for _ in range(plateau_steps):
            append_state(target, plateau_id, "plateau")
        previous = target

    return schedule


def build_repeated_state_drift_schedule(
    state_sequence=("a", "b", "c", "a", "b", "a", "b", "c"),
    initial_plateau_steps=4,
    transition_steps=2,
    plateau_steps=4,
    final_plateau_extra_steps=0,
):
    if initial_plateau_steps < 4:
        raise ValueError("initial_plateau_steps must be at least 4")
    if transition_steps < 1:
        raise ValueError("transition_steps must be positive")
    if plateau_steps < 3:
        raise ValueError("plateau_steps must be at least 3")
    if final_plateau_extra_steps < 0:
        raise ValueError(
            "final_plateau_extra_steps must be non-negative"
        )
    if len(state_sequence) < 2:
        raise ValueError("state_sequence must contain at least two states")

    state_lookup = {
        state["state_id"]: state
        for state in MULTISTAGE_PHOTOMETRIC_STATES
        if state["state_id"] in {"a", "b", "c", "d"}
    }
    unknown_states = [
        state_id
        for state_id in state_sequence
        if state_id not in state_lookup
    ]
    if unknown_states:
        raise ValueError(
            f"unknown photometric states: {sorted(set(unknown_states))}"
        )

    schedule = []
    visit_counts = {}

    def next_visit(state_id):
        visit_counts[state_id] = visit_counts.get(state_id, 0) + 1
        return visit_counts[state_id]

    def append_state(
        state,
        phase,
        segment_kind,
        state_id,
        state_visit,
        is_revisit,
        **extra,
    ):
        schedule.append(
            {
                "step": len(schedule),
                "phase": phase,
                "segment_id": phase,
                "segment_kind": segment_kind,
                "state_id": state_id,
                "state_visit": int(state_visit),
                "is_revisit": bool(is_revisit),
                "severity": _photometric_severity(state),
                "brightness": float(state["brightness"]),
                "contrast": float(state["contrast"]),
                "saturation": float(state["saturation"]),
                "hue": float(state["hue"]),
                **extra,
            }
        )

    initial_id = state_sequence[0]
    initial = state_lookup[initial_id]
    initial_visit = next_visit(initial_id)
    initial_segment = f"state_{initial_id}_visit{initial_visit}"
    for _ in range(initial_plateau_steps):
        append_state(
            initial,
            initial_segment,
            "plateau",
            initial_id,
            initial_visit,
            False,
        )

    previous_id = initial_id
    previous = initial
    previous_visit = initial_visit
    remaining_state_ids = tuple(state_sequence[1:])
    for target_index, target_id in enumerate(remaining_state_ids):
        target = state_lookup[target_id]
        target_visit = next_visit(target_id)
        target_is_revisit = target_visit > 1
        transition_id = (
            f"transition_{previous_id}_visit{previous_visit}"
            f"_to_{target_id}_visit{target_visit}"
        )
        for offset in range(1, transition_steps + 1):
            fraction = offset / float(transition_steps + 1)
            interpolated = {
                key: (
                    float(previous[key])
                    + fraction * (float(target[key]) - float(previous[key]))
                )
                for key in (
                    "brightness",
                    "contrast",
                    "saturation",
                    "hue",
                )
            }
            append_state(
                interpolated,
                transition_id,
                "transition",
                target_id,
                target_visit,
                target_is_revisit,
                from_state_id=previous_id,
                from_state_visit=int(previous_visit),
                to_state_id=target_id,
                to_state_visit=int(target_visit),
            )

        plateau_id = f"state_{target_id}_visit{target_visit}"
        target_plateau_steps = plateau_steps
        if target_index == len(remaining_state_ids) - 1:
            target_plateau_steps += final_plateau_extra_steps
        for _ in range(target_plateau_steps):
            append_state(
                target,
                plateau_id,
                "plateau",
                target_id,
                target_visit,
                target_is_revisit,
            )
        previous_id = target_id
        previous = target
        previous_visit = target_visit

    return schedule


def build_segment_balanced_stream_records(
    records,
    segment_sizes,
    anomaly_ratio,
    seed,
    allow_reuse=False,
):
    if not 0.0 < anomaly_ratio < 1.0:
        raise ValueError("anomaly_ratio must be within (0, 1)")
    if not segment_sizes or any(int(size) < 2 for size in segment_sizes):
        raise ValueError("each segment must contain at least 2 samples")

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

    segment_counts = []
    for raw_size in segment_sizes:
        size = int(raw_size)
        anomaly_count = max(
            1,
            min(size - 1, round(size * anomaly_ratio)),
        )
        segment_counts.append((size - anomaly_count, anomaly_count))

    required_normal = sum(item[0] for item in segment_counts)
    required_anomaly = sum(item[1] for item in segment_counts)
    if not normal_records:
        raise ValueError("no normal images available")
    if not anomaly_records:
        raise ValueError("no anomaly images available")
    if len(normal_records) < required_normal and not allow_reuse:
        raise ValueError(
            f"not enough normal images: need {required_normal}, "
            f"found {len(normal_records)}"
        )
    if len(anomaly_records) < required_anomaly and not allow_reuse:
        raise ValueError(
            f"not enough anomaly images: need {required_anomaly}, "
            f"found {len(anomaly_records)}"
        )

    def expand_pool(pool, required_count):
        expanded = list(pool)
        while len(expanded) < required_count:
            cycle = list(pool)
            generator.shuffle(cycle)
            expanded.extend(cycle)
        return expanded[:required_count]

    normal_records = expand_pool(normal_records, required_normal)
    anomaly_records = expand_pool(anomaly_records, required_anomaly)

    stream_records = []
    normal_offset = 0
    anomaly_offset = 0
    for normal_count, anomaly_count in segment_counts:
        segment = normal_records[
            normal_offset:normal_offset + normal_count
        ]
        segment += anomaly_records[
            anomaly_offset:anomaly_offset + anomaly_count
        ]
        normal_offset += normal_count
        anomaly_offset += anomaly_count
        generator.shuffle(segment)
        stream_records.extend(segment)
    return stream_records


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
