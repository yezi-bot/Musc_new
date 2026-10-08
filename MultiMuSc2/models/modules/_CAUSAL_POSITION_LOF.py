import math

import torch


class CausalPositionLofMemory:
    def __init__(
        self,
        k=6,
        device="cpu",
        epsilon=1.0e-6,
        position_chunk_size=64,
        image_chunk_size=8,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if position_chunk_size < 1:
            raise ValueError("position_chunk_size must be at least 1")
        if image_chunk_size < 1:
            raise ValueError("image_chunk_size must be at least 1")
        self.k = int(k)
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        self.position_chunk_size = int(position_chunk_size)
        self.image_chunk_size = int(image_chunk_size)
        self.history = []
        self._cached_key = None
        self._cached_statistics = None

    def update(self, features):
        if features.ndim != 2:
            raise ValueError("features must have shape [patch_count, dim]")
        if not torch.isfinite(features).all():
            raise ValueError("features contain non-finite values")
        self.history.append(features.detach().float().cpu().clone())
        self._cached_key = None
        self._cached_statistics = None

    def _history_tensor(self, exclude_image_id=None):
        sample_count = len(self.history)
        if exclude_image_id is not None and not (
            0 <= exclude_image_id < sample_count
        ):
            raise ValueError("exclude_image_id is outside history")

        selected = [
            features
            for image_id, features in enumerate(self.history)
            if image_id != exclude_image_id
        ]
        if len(selected) < self.k + 1:
            return None
        return torch.stack(selected)

    @staticmethod
    def _shifted_position_ids(
        grid_size,
        row_offset,
        col_offset,
        device,
    ):
        patch_ids = torch.arange(
            grid_size * grid_size,
            device=device,
        )
        rows = patch_ids // grid_size
        cols = patch_ids % grid_size
        shifted_rows = rows + row_offset
        shifted_cols = cols + col_offset
        valid = (
            (shifted_rows >= 0)
            & (shifted_rows < grid_size)
            & (shifted_cols >= 0)
            & (shifted_cols < grid_size)
        )
        shifted = (
            shifted_rows.clamp(0, grid_size - 1) * grid_size
            + shifted_cols.clamp(0, grid_size - 1)
        )
        return shifted.long(), valid

   
    def _position_offsets(
        self,
        grid_size,
        position_radius,
        device=None,
    ):
        offset_device = torch.device(device or self.device)
        return [
            (
                row_offset,
                col_offset,
                *self._shifted_position_ids(
                    grid_size,
                    row_offset,
                    col_offset,
                    offset_device,
                ),
            )
            for row_offset in range(
                -position_radius,
                position_radius + 1,
            )
            for col_offset in range(
                -position_radius,
                position_radius + 1,
            )
        ]

    @staticmethod
    def _merge_neighbors(
        best_distances,
        best_positions,
        best_images,
        candidate_distances,
        candidate_positions,
        candidate_images,
        k,
    ):
        all_distances = torch.cat(
            [best_distances, candidate_distances],
            dim=-1,
        )
        all_positions = torch.cat(
            [best_positions, candidate_positions],
            dim=-1,
        )
        all_images = torch.cat(
            [best_images, candidate_images],
            dim=-1,
        )
        best_distances, indices = torch.topk(
            all_distances,
            k=k,
            dim=-1,
            largest=False,
        )
        best_positions = torch.gather(
            all_positions,
            dim=-1,
            index=indices,
        )
        best_images = torch.gather(
            all_images,
            dim=-1,
            index=indices,
        )
        return best_distances, best_positions, best_images

    def _reference_statistics(self, history, position_radius):
        by_position = history.permute(1, 0, 2)
        patch_count, sample_count, _ = by_position.shape
        grid_size = math.isqrt(patch_count)
        if grid_size * grid_size != patch_count:
            raise ValueError("patch count must form a square grid")

        offsets = self._position_offsets(
            grid_size,
            position_radius,
            device=by_position.device,
        )
        best_distances = torch.full(
            (patch_count, sample_count, self.k),
            float("inf"),
        )
        best_positions = torch.full(
            (patch_count, sample_count, self.k),
            -1,
            dtype=torch.long,
        )
        best_images = torch.full_like(best_positions, -1)

        for start in range(
            0,
            patch_count,
            self.position_chunk_size,
        ):
            end = min(
                start + self.position_chunk_size,
                patch_count,
            )
            for query_start in range(
                0,
                sample_count,
                self.image_chunk_size,
            ):
                query_end = min(
                    query_start + self.image_chunk_size,
                    sample_count,
                )
                query = by_position[
                    start:end,
                    query_start:query_end,
                ].to(self.device)
                local_distances = best_distances[
                    start:end,
                    query_start:query_end,
                ].to(self.device)
                local_positions = best_positions[
                    start:end,
                    query_start:query_end,
                ].to(self.device)
                local_images = best_images[
                    start:end,
                    query_start:query_end,
                ].to(self.device)
                query_ids = torch.arange(
                    query_start,
                    query_end,
                    device=self.device,
                )

                for row_offset, col_offset, shifted, valid in offsets:
                    candidate_position_ids = shifted[start:end]
                    position_ids_device = candidate_position_ids.to(
                        self.device
                    )
                    valid_positions = valid[start:end].to(self.device)
                    for candidate_start in range(
                        0,
                        sample_count,
                        self.image_chunk_size,
                    ):
                        candidate_end = min(
                            candidate_start + self.image_chunk_size,
                            sample_count,
                        )
                        candidate_ids = torch.arange(
                            candidate_start,
                            candidate_end,
                            device=self.device,
                        )
                        candidates = by_position[
                            candidate_position_ids,
                            candidate_start:candidate_end,
                        ].to(self.device)
                        distances = torch.cdist(query, candidates)
                        distances = distances.masked_fill(
                            ~valid_positions[:, None, None],
                            float("inf"),
                        )
                        if row_offset == 0 and col_offset == 0:
                            same_image = (
                                query_ids[:, None]
                                == candidate_ids[None, :]
                            )
                            distances = distances.masked_fill(
                                same_image[None, :, :],
                                float("inf"),
                            )

                        position_ids = position_ids_device[
                            :, None, None
                        ].expand(
                            end - start,
                            query_end - query_start,
                            candidate_end - candidate_start,
                        )
                        image_ids = candidate_ids[
                            None, None, :
                        ].expand_as(position_ids)
                        (
                            local_distances,
                            local_positions,
                            local_images,
                        ) = self._merge_neighbors(
                            local_distances,
                            local_positions,
                            local_images,
                            distances,
                            position_ids,
                            image_ids,
                            self.k,
                        )

                best_distances[
                    start:end,
                    query_start:query_end,
                ] = local_distances.cpu()
                best_positions[
                    start:end,
                    query_start:query_end,
                ] = local_positions.cpu()
                best_images[
                    start:end,
                    query_start:query_end,
                ] = local_images.cpu()

        if not torch.isfinite(best_distances).all():
            raise RuntimeError("reference neighborhood has fewer than k points")

        k_distance = best_distances[..., -1]
        neighbor_k_distance = k_distance[
            best_positions,
            best_images,
        ]
        reachability = torch.maximum(
            best_distances,
            neighbor_k_distance,
        )
        lrd = 1.0 / (
            reachability.mean(dim=-1) + self.epsilon
        )
        return by_position, k_distance, lrd

    def _statistics(self, position_radius, exclude_image_id):
        key = (
            len(self.history),
            int(position_radius),
            exclude_image_id,
        )
        if self._cached_key == key:
            return self._cached_statistics

        history = self._history_tensor(exclude_image_id)
        if history is None:
            return None
        statistics = self._reference_statistics(
            history,
            position_radius,
        )
        self._cached_key = key
        self._cached_statistics = statistics
        return statistics

    def score(
        self,
        features,
        position_radius=0,
        exclude_image_id=None,
    ):
        if features.ndim != 2:
            raise ValueError("features must have shape [patch_count, dim]")
        if position_radius < 0:
            raise ValueError("position_radius must be non-negative")

        statistics = self._statistics(
            position_radius,
            exclude_image_id,
        )
        if statistics is None:
            return None
        by_position, k_distance, lrd = statistics
        if features.shape != (
            by_position.shape[0],
            by_position.shape[2],
        ):
            raise ValueError("query shape does not match history")

        patch_count, sample_count, _ = by_position.shape
        grid_size = math.isqrt(patch_count)
        offsets = self._position_offsets(
            grid_size,
            position_radius,
            device=by_position.device,
        )
        query = features.detach().float().cpu()
        lof = torch.empty(patch_count)

        for start in range(
            0,
            patch_count,
            self.position_chunk_size,
        ):
            end = min(
                start + self.position_chunk_size,
                patch_count,
            )
            chunk_size = end - start
            best_distances = torch.full(
                (chunk_size, self.k),
                float("inf"),
                device=self.device,
            )
            best_positions = torch.full(
                (chunk_size, self.k),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            best_images = torch.full_like(best_positions, -1)
            query_chunk = query[start:end].to(self.device)

            for _, _, shifted, valid in offsets:
                candidate_position_ids = shifted[start:end]
                position_ids_device = candidate_position_ids.to(
                    self.device
                )
                valid_positions = valid[start:end].to(self.device)
                for candidate_start in range(
                    0,
                    sample_count,
                    self.image_chunk_size,
                ):
                    candidate_end = min(
                        candidate_start + self.image_chunk_size,
                        sample_count,
                    )
                    candidate_ids = torch.arange(
                        candidate_start,
                        candidate_end,
                        device=self.device,
                    )
                    candidates = by_position[
                        candidate_position_ids,
                        candidate_start:candidate_end,
                    ].to(self.device)
                    distances = torch.cdist(
                        query_chunk[:, None, :],
                        candidates,
                    ).squeeze(1)
                    distances = distances.masked_fill(
                        ~valid_positions[:, None],
                        float("inf"),
                    )
                    candidate_positions = position_ids_device[
                        :, None
                    ].expand(
                        -1,
                        candidate_end - candidate_start,
                    )
                    candidate_images = candidate_ids[None, :].expand(
                        chunk_size,
                        -1,
                    )
                    (
                        best_distances,
                        best_positions,
                        best_images,
                    ) = self._merge_neighbors(
                        best_distances,
                        best_positions,
                        best_images,
                        distances,
                        candidate_positions,
                        candidate_images,
                        self.k,
                    )

            if not torch.isfinite(best_distances).all():
                raise RuntimeError("query neighborhood has fewer than k points")
            best_distances = best_distances.cpu()
            best_positions = best_positions.cpu()
            best_images = best_images.cpu()
            selected_k_distances = k_distance[
                best_positions,
                best_images,
            ]
            selected_lrds = lrd[
                best_positions,
                best_images,
            ]
            reachability = torch.maximum(
                best_distances,
                selected_k_distances,
            )
            query_lrd = 1.0 / (
                reachability.mean(dim=-1) + self.epsilon
            )
            lof[start:end] = (
                selected_lrds.mean(dim=-1) / query_lrd
            )
        return lof


def lof_tail_mean(lof_scores, tail_fraction=0.15):
    if lof_scores is None:
        return None
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be within (0, 1]")
    if lof_scores.ndim != 1 or lof_scores.numel() == 0:
        raise ValueError("lof_scores must be a non-empty vector")
    count = max(1, int(math.ceil(lof_scores.numel() * tail_fraction)))
    return float(
        torch.topk(lof_scores.float(), k=count).values.mean()
    )


def lof_rank_reliability(lof_scores, minimum=0.1):
    if lof_scores is None:
        return None
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum must be within (0, 1]")
    order = torch.argsort(lof_scores)
    ranks = torch.empty_like(lof_scores, dtype=torch.float32)
    ranks[order] = torch.arange(
        lof_scores.numel(),
        dtype=torch.float32,
    )
    reliability = (
        lof_scores.numel() - ranks
    ) / max(lof_scores.numel(), 1)
    return reliability.clamp(min=minimum, max=1.0)