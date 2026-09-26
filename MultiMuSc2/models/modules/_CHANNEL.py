import torch
import torch.nn.functional as F

class Channel:
    def __init__(self, features,image_id,ttl=5):
        # [NLC] C
        self.center = features.detach().clone() 
        self.features = [features.detach().clone()]
        self.image_id = [image_id]
        self.span = 1
        self.ttl = ttl

class ChannelMemory:
    def __init__(self, max_ttl=5):
       self.channels = []
       self.max_ttl = max_ttl      

    def initialize(self, features, image_id):
        for patch_feature in features:
         self.channel = Channel(features=patch_feature,image_id=image_id,ttl=self.max_ttl)
         self.channels.append(self.channel) 

    def update(self, features, image_id):

     centers = torch.stack(
         [channel.center for channel in self.channels],
         dim=0
     )
    # 每个patch到center的距离
     dist = torch.cdist(features, centers) 
    # 最近邻
     patch_to_channel = dist.argmin(dim=1)
     channel_to_patch = dist.argmin(dim=0)
     
     matched_patches = set()
     matched_num = 0

     for patch_id in range(features.shape[0]):
        # 最近cahnnel的编号
        channel_id = patch_to_channel[patch_id].item()
        if channel_to_patch[channel_id].item() == patch_id:

             channel = self.channels[channel_id]
 
             channel.features.append(
                 features[patch_id].detach().clone()
             )

             channel.image_id.append(image_id)

             channel.span += 1

             # 暂时用所有历史feature均值更新center
             channel.center = torch.stack(
                 channel.features,
                 dim=0
             ).mean(dim=0)

             matched_patches.add(patch_id)
             matched_num += 1
     new_num=0
     for patch_id in range(features.shape[0]):
         if patch_id not in matched_patches:
             new_channel = Channel(
                 features=features[patch_id].detach().clone(),
                 image_id=image_id,
                 ttl=self.max_ttl
             )
             self.channels.append(new_channel)
             new_num += 1

     print(
         "image:",
         image_id,
         "mutual matched:",
         matched_num,
         "new channels:",
         new_num,
         "total channels:",
         len(self.channels)
     )
