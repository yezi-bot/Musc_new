import torch

class Channel:
    def __init__(self, feature,image_id,patch_id,ttl,reliability):
        # [NLC] C
        feature = feature.detach().cpu().clone()
        self.seed = feature
        self.latest_feature = feature

        self.seed_image_id = int(image_id)
        self.seed_patch_id = int(patch_id)
        self.latest_image_id = int(image_id)
        self.latest_patch_id = int(patch_id)

        self.span = 1
        self.effective_span = float(reliability)
        self.ttl = int(ttl)

    def update(self, feature, image_id, patch_id, ttl, reliability):
        self.latest_feature = feature.detach().cpu().clone()
        self.latest_image_id = int(image_id)
        self.latest_patch_id = int(patch_id)

        self.span += 1
        self.effective_span += float(reliability)
        self.ttl = int(ttl)    


class ChannelMemory:
    def __init__(self, 
        max_ttl=5,
        mature_span=3.0,
        density_k=5,
        device="cpu",
        position_radius=1,):
       if max_ttl < 1:
            raise ValueError("max_ttl must be at least 1")
       if mature_span <= 0:
            raise ValueError("mature_span must be positive")
       if density_k < 1:
            raise ValueError("density_k must be at least 1")
       if position_radius < 1:
            raise ValueError("position_radius must be at least 1")      

       self.channels = []
       self.max_ttl = int(max_ttl)
       self.mature_span = float(mature_span)
       self.density_k = int(density_k)
       self.device = torch.device(device)
       self.position_radius = int(position_radius)

       self.deleted_last_step = 0
       self.matched_last_step = 0
       self.created_last_step = 0


    @staticmethod
    def _validate_features(features, grid_size):
        if features.ndim != 2:
            raise ValueError("features must have shape [patch_count, feature_dim]")
        if features.shape[0] < 2:
            raise ValueError("at least two patches are required")
        if grid_size * grid_size != features.shape[0]:
            raise ValueError(
                "grid_size squared must equal the number of patch features"
            )
        if not torch.isfinite(features).all():
            raise ValueError("features contain non-finite values")

    #位置掩码
    def _position_mask(self, patch_count, channel_patch_ids, grid_size):
        patch_ids = torch.arange(patch_count, device=self.device)
        channel_patch_ids = torch.as_tensor(
            channel_patch_ids,
            device=self.device,
            dtype=torch.long,
        )

        patch_rows = patch_ids[:, None] // grid_size
        patch_cols = patch_ids[:, None] % grid_size
        channel_rows = channel_patch_ids[None, :] // grid_size
        channel_cols = channel_patch_ids[None, :] % grid_size

        valid = (
            (patch_rows - channel_rows).abs() <= self.position_radius
        )
        valid &= (
            (patch_cols - channel_cols).abs() <= self.position_radius
        )
        return valid


    def mature_channels(self):
        return [
            channel
            for channel in self.channels
            if channel.effective_span >= self.mature_span
        ]

    def association_profile(self, features, grid_size, current_image_id=None):
        self._validate_features(features, grid_size)
        patch_count = features.shape[0]
        eligible = [
            (index, channel)
            for index, channel in enumerate(self.channels)
            if channel.ttl > 1
        ]
        if not eligible:
            return {
                "matched_fraction": 0.0,
                "mature_match_fraction": 0.0,
                "provisional_match_fraction": 0.0,
                "recent_provisional_match_fraction": 0.0,
                "matched_channel_ids": [],
                "provisional_channel_ids": [],
            }

        current = features.detach().float().to(self.device)
        candidates = torch.stack(
            [channel.seed for _, channel in eligible]
        ).float().to(self.device)
        distances = torch.cdist(current, candidates)
        valid = self._position_mask(
            patch_count,
            [channel.latest_patch_id for _, channel in eligible],
            grid_size,
        )
        distances = distances.masked_fill(~valid, float("inf"))
        patch_to_channel = distances.argmin(dim=1)
        channel_to_patch = distances.argmin(dim=0)

        matched_channel_ids = []
        provisional_channel_ids = []
        recent_provisional_count = 0
        mature_count = 0
        for patch_id in range(patch_count):
            local_channel_id = int(patch_to_channel[patch_id])
            if not torch.isfinite(distances[patch_id, local_channel_id]):
                continue
            if int(channel_to_patch[local_channel_id]) != patch_id:
                continue
            channel_id, channel = eligible[local_channel_id]
            matched_channel_ids.append(channel_id)
            if channel.effective_span >= self.mature_span:
                mature_count += 1
            else:
                provisional_channel_ids.append(channel_id)
                if (
                    current_image_id is not None
                    and channel.seed_image_id == current_image_id - 1
                ):
                    recent_provisional_count += 1

        matched_count = len(matched_channel_ids)
        provisional_count = len(provisional_channel_ids)
        return {
            "matched_fraction": matched_count / patch_count,
            "mature_match_fraction": mature_count / patch_count,
            "provisional_match_fraction": provisional_count / patch_count,
            "recent_provisional_match_fraction": (
                recent_provisional_count / patch_count
            ),
            "matched_channel_ids": matched_channel_ids,
            "provisional_channel_ids": provisional_channel_ids,
        }


    def patch_to_mature_distances(self, features, grid_size):
        self._validate_features(features, grid_size)
        mature = self.mature_channels()

        if not mature:
            return None

        current = features.detach().float().to(self.device)
        seeds = torch.stack(
            [channel.seed for channel in mature]
        ).float().to(self.device)

        distances = torch.cdist(current, seeds)
        valid = self._position_mask(
            current.shape[0],
            # 最近的一次channel_id
            [channel.latest_patch_id for channel in mature],
            grid_size,
        )
        # 不合法距离设置为无穷大
        distances = distances.masked_fill(~valid, float("inf"))
        return distances.amin(dim=1).cpu()


    def patch_reliability(self, features, grid_size):
        self._validate_features(features, grid_size)
        current = features.detach().float().to(self.device)

        if self.channels:
            # Channel里面的最新特征
            candidates = torch.stack(
                [channel.latest_feature for channel in self.channels]
            ).float().to(self.device)

            distances = torch.cdist(current, candidates)
            valid = self._position_mask(
                current.shape[0],
                [channel.latest_patch_id for channel in self.channels],
                grid_size,
            )
            distances = distances.masked_fill(~valid, float("inf"))
        else:
            distances = torch.cdist(current, current)
            valid = self._position_mask(
                current.shape[0],
                torch.arange(current.shape[0]),
                grid_size,
            )
            valid.fill_diagonal_(False)
            distances = distances.masked_fill(~valid, float("inf"))
        
        # 初始化张量存每个patch的分数，[patch cout,feature dim]
        scores = torch.empty(current.shape[0], device=self.device)

        for patch_id in range(current.shape[0]):
            # 这个 patch 到所有候选的距离
            finite = distances[patch_id][
                torch.isfinite(distances[patch_id])
            ]

            if finite.numel() == 0:
                current_distance = torch.cdist(
                    current[patch_id : patch_id + 1],
                    current,
                )[0]

                local_valid = self._position_mask(
                  current.shape[0],
                  torch.arange(current.shape[0], device=self.device),
                  grid_size,
                 )[patch_id].clone()
                local_valid[patch_id]=False
                finite =current_distance[local_valid]
                if finite.numel() == 0:
                 raise RuntimeError(
                 f"patch {patch_id} has no valid local neighbour "
                 f"within position_radius={self.position_radius}" )

            k = min(self.density_k, finite.numel())
            scores[patch_id] = torch.topk(
                finite,
                k=k,
                largest=False,
            ).values.mean()
        
        # 返回排序后索引默认升序
        order = torch.argsort(scores)
        ranks = torch.empty_like(scores)
        # 按order顺序填入
        ranks[order] = torch.arange(
            scores.numel(),
            device=self.device,
            dtype=scores.dtype,
        )
        # 排名归一化，reliability越小越靠前
        reliability = 1.0 - ranks / max(scores.numel() - 1, 1)
        # 限制范围
        reliability = reliability.clamp(min=0.1, max=1.0)

        return scores.cpu(), reliability.cpu()

    def update(
        self,
        features,
        image_id,
        grid_size,
        patch_reliabilities,
    ):
        self._validate_features(features, grid_size)

        reliabilities = torch.as_tensor(
            patch_reliabilities,
            dtype=torch.float32,
        ).cpu()
        
        # 每个patch有一个可靠性
        if reliabilities.shape != (features.shape[0],):
            raise ValueError(
                "patch_reliabilities must contain one value per patch"
            )
        if not torch.isfinite(reliabilities).all():
            raise ValueError("patch_reliabilities contain non-finite values")
        if ((reliabilities <= 0) | (reliabilities > 1)).any():
            raise ValueError(
                "patch_reliabilities must be in the interval (0, 1]"
            )

        for channel in self.channels:
            channel.ttl -= 1

        previous_count = len(self.channels)
        # 删除过期channel
        self.channels = [
            channel for channel in self.channels if channel.ttl > 0
        ]

        self.deleted_last_step = previous_count - len(self.channels)
        self.matched_last_step = 0
        self.created_last_step = 0

        if not self.channels:
            for patch_id, feature in enumerate(features):
                self.channels.append(
                    Channel(
                        feature=feature,
                        image_id=image_id,
                        patch_id=patch_id,
                        ttl=self.max_ttl,
                        reliability=reliabilities[patch_id],
                    )
                )
            self.created_last_step = features.shape[0]
            return
        
        current = features.detach().float().to(self.device)
        seeds = torch.stack(
            [channel.seed for channel in self.channels]
        ).float().to(self.device)

        # 计算当前patch到channel距离 [L,M]
        distances = torch.cdist(current, seeds)
        valid = self._position_mask(
            current.shape[0],
            [channel.latest_patch_id for channel in self.channels],
            grid_size,
        )
        distances = distances.masked_fill(~valid, float("inf"))

        patch_to_channel = distances.argmin(dim=1)
        channel_to_patch = distances.argmin(dim=0)
        matched_patches = set()

        for patch_id in range(current.shape[0]):
            channel_id = int(patch_to_channel[patch_id])

            if not torch.isfinite(distances[patch_id, channel_id]):
                continue
            if int(channel_to_patch[channel_id]) != patch_id:
                continue

            # 匹配成功的channel
            self.channels[channel_id].update(
                feature=features[patch_id],
                image_id=image_id,
                patch_id=patch_id,
                ttl=self.max_ttl,
                reliability=reliabilities[patch_id],
            )
            matched_patches.add(patch_id)
            self.matched_last_step += 1
        # 没匹配成功
        for patch_id, feature in enumerate(features):
            if patch_id in matched_patches:
                continue

            self.channels.append(
                Channel(
                    feature=feature,
                    image_id=image_id,
                    patch_id=patch_id,
                    ttl=self.max_ttl,
                    reliability=reliabilities[patch_id],
                )
            )
            self.created_last_step += 1
