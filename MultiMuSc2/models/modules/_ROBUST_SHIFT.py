import statistics

import torch


def _validate_patch_ids(
    name,
    patch_ids,
    patch_count,
    device,
    allow_duplicates=False,
):
    ids = torch.as_tensor(patch_ids, dtype=torch.long, device=device)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if ((ids < 0) | (ids >= patch_count)).any():
        raise ValueError(f"{name} contains an invalid patch index")
    if not allow_duplicates and ids.unique().numel() != ids.numel():
        raise ValueError(f"{name} must not contain duplicates")
    return ids


def _position_mask(target_ids, reference_ids, grid_size, position_radius):
    if position_radius is None:
        return torch.ones(
            (target_ids.numel(), reference_ids.numel()),
            dtype=torch.bool,
            device=target_ids.device,
        )
    target_rows = torch.div(target_ids, grid_size, rounding_mode="floor")
    target_cols = target_ids % grid_size
    reference_rows = torch.div(
        reference_ids,
        grid_size,
        rounding_mode="floor",
    )
    reference_cols = reference_ids % grid_size
    return (
        (target_rows[:, None] - reference_rows[None, :]).abs()
        <= position_radius
    ) & (
        (target_cols[:, None] - reference_cols[None, :]).abs()
        <= position_radius
    )


def robust_sinkhorn_profile(
    target_features,
    reference_features,
    target_patch_ids,
    reference_patch_ids,
    grid_size,
    position_radius=1,
    epsilon=0.05,
    iterations=20,
    allow_duplicate_patch_ids=False,
):
    if target_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("features must have shape [patch, feature]")
    if target_features.shape[1] != reference_features.shape[1]:
        raise ValueError("target and reference feature dimensions must match")
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if position_radius is not None and position_radius < 0:
        raise ValueError("position_radius must be non-negative")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    patch_count = grid_size * grid_size
    target_ids = _validate_patch_ids(
        "target_patch_ids",
        target_patch_ids,
        patch_count,
        target_features.device,
        allow_duplicates=allow_duplicate_patch_ids,
    )
    reference_ids = _validate_patch_ids(
        "reference_patch_ids",
        reference_patch_ids,
        patch_count,
        target_features.device,
        allow_duplicates=allow_duplicate_patch_ids,
    )
    if target_features.shape[0] != target_ids.numel():
        raise ValueError("target_patch_ids must match target feature count")
    if reference_features.shape[0] != reference_ids.numel():
        raise ValueError("reference_patch_ids must match reference feature count")

    target = target_features.detach().float()
    reference = reference_features.detach().float().to(target.device)
    if not torch.isfinite(target).all() or not torch.isfinite(reference).all():
        raise ValueError("features contain non-finite values")

    costs = torch.cdist(target, reference)
    valid = _position_mask(
        target_ids,
        reference_ids,
        grid_size,
        position_radius,
    )
    if not valid.any(dim=1).all() or not valid.any(dim=0).all():
        raise ValueError("position constraint leaves a patch without a match")

    finite_costs = costs[valid]
    cost_scale = finite_costs.median().clamp_min(1.0e-6)
    normalized = costs / cost_scale
    kernel = torch.exp(-normalized / epsilon) * valid
    kernel = kernel.clamp_min(1.0e-30)

    target_mass = torch.full(
        (target.shape[0],),
        1.0 / target.shape[0],
        device=target.device,
    )
    reference_mass = torch.full(
        (reference.shape[0],),
        1.0 / reference.shape[0],
        device=target.device,
    )
    left = torch.ones_like(target_mass)
    right = torch.ones_like(reference_mass)
    for _ in range(iterations):
        left = target_mass / (kernel @ right).clamp_min(1.0e-30)
        right = reference_mass / (
            kernel.transpose(0, 1) @ left
        ).clamp_min(1.0e-30)

    plan = left[:, None] * kernel * right[None, :]
    plan = plan * valid
    plan = plan / plan.sum().clamp_min(1.0e-30)
    sinkhorn_cost = float((plan * costs).sum().cpu())

    row_best = plan.argmax(dim=1)
    column_best = plan.argmax(dim=0)
    row_ids = torch.arange(target.shape[0], device=target.device)
    column_ids = torch.arange(reference.shape[0], device=target.device)
    discrete = torch.zeros_like(valid)
    discrete[row_ids, row_best] = True
    discrete[column_best, column_ids] = True
    discrete &= valid
    discrete_cost = float(costs[discrete].mean().cpu())

    mutual = column_best[row_best] == row_ids
    mutual_fraction = float(mutual.float().mean().cpu())
    return {
        "sinkhorn_cost": sinkhorn_cost,
        "discrete_cost": discrete_cost,
        "mutual_fraction": mutual_fraction,
        "target_patch_count": int(target.shape[0]),
        "reference_patch_count": int(reference.shape[0]),
    }


def robust_window_sinkhorn_profile(
    target_feature_sets,
    reference_feature_sets,
    patch_ids,
    grid_size,
    position_radius=1,
    epsilon=0.05,
    iterations=20,
):
    target_sets = torch.as_tensor(target_feature_sets)
    reference_sets = torch.as_tensor(reference_feature_sets)
    if target_sets.ndim == 2:
        target_sets = target_sets.unsqueeze(0)
    if reference_sets.ndim == 2:
        reference_sets = reference_sets.unsqueeze(0)
    if target_sets.ndim != 3 or reference_sets.ndim != 3:
        raise ValueError(
            "window features must have shape [image, patch, feature]"
        )
    if target_sets.shape[0] == 0 or reference_sets.shape[0] == 0:
        raise ValueError("window features must contain at least one image")
    if target_sets.shape[1:] != reference_sets.shape[1:]:
        raise ValueError("target and reference window shapes must match")
    reference_sets = reference_sets.to(target_sets.device)

    patch_count = grid_size * grid_size
    sampled_ids = _validate_patch_ids(
        "patch_ids",
        patch_ids,
        patch_count,
        target_sets.device,
    )
    if target_sets.shape[1] != patch_count:
        raise ValueError("window patch count must match grid_size")

    pair_profiles = []
    for target_index in range(target_sets.shape[0]):
        for reference_index in range(reference_sets.shape[0]):
            pair_profile = robust_sinkhorn_profile(
                target_sets[target_index, sampled_ids],
                reference_sets[reference_index, sampled_ids],
                target_patch_ids=sampled_ids,
                reference_patch_ids=sampled_ids,
                grid_size=grid_size,
                position_radius=position_radius,
                epsilon=epsilon,
                iterations=iterations,
            )
            pair_profiles.append(
                {
                    "target_image_index": int(target_index),
                    "reference_image_index": int(reference_index),
                    **pair_profile,
                }
            )

    discrete_costs = [
        profile["discrete_cost"] for profile in pair_profiles
    ]
    sinkhorn_costs = [
        profile["sinkhorn_cost"] for profile in pair_profiles
    ]
    mutual_fractions = [
        profile["mutual_fraction"] for profile in pair_profiles
    ]
    return {
        "sinkhorn_cost": float(statistics.median(sinkhorn_costs)),
        "sinkhorn_cost_max": float(max(sinkhorn_costs)),
        "discrete_cost": float(statistics.median(discrete_costs)),
        "discrete_cost_max": float(max(discrete_costs)),
        "mutual_fraction": float(statistics.fmean(mutual_fractions)),
        "mutual_fraction_min": float(min(mutual_fractions)),
        "target_patch_count": int(
            target_sets.shape[0] * sampled_ids.numel()
        ),
        "reference_patch_count": int(
            reference_sets.shape[0] * sampled_ids.numel()
        ),
        "patch_count_per_image": int(sampled_ids.numel()),
        "target_image_count": int(target_sets.shape[0]),
        "reference_image_count": int(reference_sets.shape[0]),
        "pair_count": len(pair_profiles),
        "pair_profiles": pair_profiles,
    }


def pairwise_window_sinkhorn_profile(
    feature_sets,
    patch_ids,
    grid_size,
    position_radius=1,
    epsilon=0.05,
    iterations=20,
):
    sets = torch.as_tensor(feature_sets)
    if sets.ndim != 3:
        raise ValueError(
            "window features must have shape [image, patch, feature]"
        )
    if sets.shape[0] < 2:
        raise ValueError("pairwise window requires at least two images")

    patch_count = grid_size * grid_size
    sampled_ids = _validate_patch_ids(
        "patch_ids",
        patch_ids,
        patch_count,
        sets.device,
    )
    if sets.shape[1] != patch_count:
        raise ValueError("window patch count must match grid_size")

    pair_profiles = []
    for left_index in range(sets.shape[0] - 1):
        for right_index in range(left_index + 1, sets.shape[0]):
            pair_profile = robust_sinkhorn_profile(
                sets[left_index, sampled_ids],
                sets[right_index, sampled_ids],
                target_patch_ids=sampled_ids,
                reference_patch_ids=sampled_ids,
                grid_size=grid_size,
                position_radius=position_radius,
                epsilon=epsilon,
                iterations=iterations,
            )
            pair_profiles.append(
                {
                    "left_image_index": int(left_index),
                    "right_image_index": int(right_index),
                    **pair_profile,
                }
            )

    discrete_costs = [
        profile["discrete_cost"] for profile in pair_profiles
    ]
    sinkhorn_costs = [
        profile["sinkhorn_cost"] for profile in pair_profiles
    ]
    mutual_fractions = [
        profile["mutual_fraction"] for profile in pair_profiles
    ]
    return {
        "discrete_cost_median": float(statistics.median(discrete_costs)),
        "discrete_cost_max": float(max(discrete_costs)),
        "sinkhorn_cost_median": float(statistics.median(sinkhorn_costs)),
        "sinkhorn_cost_max": float(max(sinkhorn_costs)),
        "mutual_fraction_mean": float(statistics.fmean(mutual_fractions)),
        "mutual_fraction_min": float(min(mutual_fractions)),
        "image_count": int(sets.shape[0]),
        "pair_count": len(pair_profiles),
        "pair_profiles": pair_profiles,
    }


def uniformly_sample_patch_ids(patch_count, max_patches, device):
    if patch_count < 1:
        raise ValueError("patch_count must be positive")
    if max_patches < 1:
        raise ValueError("max_patches must be positive")
    keep = min(int(max_patches), int(patch_count))
    if keep == patch_count:
        return torch.arange(patch_count, device=device)
    positions = torch.linspace(
        0,
        patch_count - 1,
        steps=keep,
        device=device,
    )
    ids = positions.round().long().unique()
    if ids.numel() != keep:
        raise RuntimeError("uniform patch sampling produced duplicate indices")
    return ids
