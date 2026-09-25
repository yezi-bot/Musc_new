import torch
from sklearn import linear_model

patch2image = torch.randn(1,37,37)
patch2image2 = torch.randn(1,37,37)
patch2image_fusion = torch.cat((patch2image,patch2image2),dim=0)
patch2image_fusion=patch2image_fusion.reshape(2,37*37).T
imagenum=20
s_maptrain = torch.randn(1369*imagenum,2)
detect_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=0.5,  max_iter=1000)
s_map1 = torch.cat([patch2image, patch2image2], dim=1)#.squeeze().reshape(3, -1).permute(1, 0)
detect_fuser.fit(s_maptrain)
s_map2 = torch.tensor(detect_fuser.score_samples(patch2image_fusion))
print(s_map2.shape)