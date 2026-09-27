import torch
import torch.nn.functional as F

class Channel:
    def __init__(self, features,image_id,patch_id,ttl=5):
        # [NLC] C
        self.seed = features.detach().clone() 
        self.features = [features.detach().clone()]
        self.image_id = [image_id]
        self.patch_id=[patch_id]
        self.span = 1
        self.ttl = ttl

class ChannelMemory:
    def __init__(self, max_ttl=5):
       self.channels = []
       self.max_ttl = max_ttl      

    def initialize(self, features, image_id):
        for patch_id, patch_feature in enumerate(features):
         self.channel = Channel(features=patch_feature,image_id=image_id,patch_id=patch_id,ttl=self.max_ttl)
         self.channels.append(self.channel) 

    def update(self, features, image_id):
     
     for channel in self.channels:
         channel.ttl -= 1
     before_len = len(self.channels)    
     self.channels = [channel for channel in self.channels if channel.ttl > 0]
     after_len = len(self.channels)
     deleted_num = before_len - after_len
     if len(self.channels) == 0:
         self.initialize(features, image_id)
         return
     seeds = torch.stack(
         [channel.seed for channel in self.channels],
         dim=0
     )
    # 每个patch到seed的距离
     dist = torch.cdist(features, seeds) 
    # 最近邻
     patch_to_channel = dist.argmin(dim=1)
     channel_to_patch = dist.argmin(dim=0)
     
     matched_patches = set()

     for patch_id in range(features.shape[0]):
        # 最近channel的编号
        channel_id = patch_to_channel[patch_id].item()
        if channel_to_patch[channel_id].item() == patch_id:

             channel = self.channels[channel_id]
 
             channel.features.append(
                 features[patch_id].detach().clone()
             )

             channel.image_id.append(image_id)
             channel.patch_id.append(patch_id)

             channel.span += 1
             channel.ttl = self.max_ttl
             matched_patches.add(patch_id)
             
     for patch_id in range(features.shape[0]):
         if patch_id not in matched_patches:
             new_channel = Channel(
                 features=features[patch_id].detach().clone(),
                 image_id=image_id,
                 patch_id=patch_id,
                 ttl=self.max_ttl
             )
             self.channels.append(new_channel)
     spans = [c.span for c in self.channels]
     ttls = [c.ttl for c in self.channels]

     print(
        f"image {image_id}: "
        f"channels={len(self.channels)}, "
        f"max_span={max(spans)}, "
        f"min_ttl={min(ttls)}, "
        f"max_ttl={max(ttls)}"
        f"deleted_channels={deleted_num}"
     )

    def get_mature_features(self, min_span=3):
        mature_features = []
        for channel in self.channels:
            if channel.span >= min_span:
                mature_features.append(channel.seed)
        if len(mature_features) == 0:
            return None
        mature_features= torch.stack(mature_features, dim=0)
        return mature_features

    def compute_knn_density(self, k=5,span_threshold=3):
        mature_channels = [
            c for c in self.channels
            if c.span >= span_threshold
        ]
        if len(mature_channels) <= 1:
            return mature_channels,None
        seeds = torch.stack([c.seed for c in mature_channels], dim=0)
        dist = torch.cdist(seeds, seeds)
        dist.fill_diagonal_(float('inf'))  # 避免自我匹配
        k = min(k, dist.shape[1]-1)  # 确保 k 不超过可用的邻居数量
        # k个最近的patch
        knn_distances, _ = torch.topk(dist, k=k, dim=1, largest=False)
         # K 个最近距离取平均
        knn_density = knn_distances.mean(dim=1)
        return mature_channels,knn_density
   
  
