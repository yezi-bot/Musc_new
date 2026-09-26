from sympy import print_gtk
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import linear_model

sys.path.append('./models/backbone')

import datasets.mvtec as mvtec
from datasets.mvtec import _CLASSNAMES as _CLASSNAMES_mvtec_ad
import datasets.visa as visa
from datasets.visa import _CLASSNAMES as _CLASSNAMES_visa
import datasets.btad as btad
from datasets.btad import _CLASSNAMES as _CLASSNAMES_btad

import models.backbone.open_clip as open_clip
import models.backbone._backbones as _backbones
from models.modules._LNAMD import LNAMD
from models.modules._MSM import MSM,MSM2
from models.modules._RsCIN import RsCIN
from models.modules._CHANNEL import Channel, ChannelMemory
# from models.modules._CHANNEL import _CHANNEL
from utils.metrics import compute_metrics
from openpyxl import Workbook
from tqdm import tqdm
import pickle
import time
import cv2

import warnings
warnings.filterwarnings("ignore")


class MuSc():
    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.seed = seed
        self.device = torch.device("cuda:{}".format(cfg['device']) if torch.cuda.is_available() else "cpu")

        self.path = cfg['datasets']['data_path']
        self.dataset = cfg['datasets']['dataset_name']
        self.vis = cfg['testing']['vis']
        self.vis_type = cfg['testing']['vis_type']
        self.save_excel = cfg['testing']['save_excel']
        # the categories to be tested
        self.categories = cfg['datasets']['class_name']
        if isinstance(self.categories, str):
            if self.categories.lower() == 'all':
                if self.dataset == 'visa':
                    self.categories = _CLASSNAMES_visa
                elif self.dataset == 'mvtec_ad':
                    self.categories = _CLASSNAMES_mvtec_ad
                elif self.dataset == 'btad':
                    self.categories = _CLASSNAMES_btad
            else:
                self.categories = [self.categories]

        self. model_name1 = cfg['models']['backbone_name1']
        self.model_name2 = cfg['models']['backbone_name2']
        self.image_size = cfg['datasets']['img_resize']
        self.batch_size = cfg['models']['batch_size']
        self.use_dynamic_model = cfg['models']['use_dynamic_model']
        self.pretrained = cfg['models']['pretrained']
        self.features_list = [l+1 for l in cfg['models']['feature_layers']]
        self.divide_num = cfg['datasets']['divide_num']
        self.r_list = cfg['models']['r_list']
        self.r_list2 = cfg['models']['r_list2']
        self.output_dir = os.path.join(cfg['testing']['output_dir'], self.dataset, self.model_name1, 'imagesize{}'.format(self.image_size))
        os.makedirs(self.output_dir, exist_ok=True)
        self.load_backbone()
        self.input_proj = torch.nn.Linear(1024,768).to(self.device)
        self.input_proj2 = torch.nn.Linear(768,1024).to(self.device)
        self.detect_fuser = linear_model.SGDOneClassSVM(random_state=42, nu=0.5,  max_iter=1000)

    def load_backbone(self):

        # #dinov2
        self.dino_model = _backbones.load(self.model_name1)
        # 加载到指定设备
        self.dino_model.to(self.device)
        self.preprocess = None
        # clip
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name2, self.image_size, pretrained=self.pretrained)
        # preprocess按照clip要求处理
        self.clip_model.to(self.device)



    def load_datasets(self, category, divide_num=1, divide_iter=0):
        # dataloader
        if self.dataset == 'visa':
            test_dataset = visa.VisaDataset(source=self.path, split=visa.DatasetSplit.TEST,
                                            classname=category, resize=self.image_size, imagesize=self.image_size, clip_transformer=self.preprocess,
                                                divide_num=divide_num, divide_iter=divide_iter, random_seed=self.seed)
        elif self.dataset == 'mvtec_ad':
            test_dataset = mvtec.MVTecDataset(source=self.path, split=mvtec.DatasetSplit.TEST,
                                            classname=category, resize=self.image_size, imagesize=self.image_size, clip_transformer=self.preprocess,
                                                divide_num=divide_num, divide_iter=divide_iter, random_seed=self.seed)
        elif self.dataset == 'btad':
            test_dataset = btad.BTADDataset(source=self.path, split=btad.DatasetSplit.TEST,
                                            classname=category, resize=self.image_size, imagesize=self.image_size, clip_transformer=self.preprocess,
                                                divide_num=divide_num, divide_iter=divide_iter, random_seed=self.seed)
        return test_dataset

    # def visualization_seg(self, image_path_list, gt_list, pr_px, category):
    #     # 定义异常区域占比（如5%的高分区域视为异常）
    #     anomaly_percent = 5  # 可调整参数
    #
    #     if self.vis_type == 'single_norm':
    #         # 每张图单独处理
    #         for i, path in enumerate(image_path_list):
    #             anomaly_type = path.split('/')[-2]
    #             img_name = path.split('/')[-1]
    #             save_path = os.path.join(self.output_dir, category, anomaly_type)
    #             os.makedirs(save_path, exist_ok=True)
    #             save_path = os.path.join(save_path, img_name)
    #             # 获取当前图片的异常得分图
    #             score_map = pr_px[i].squeeze()
    #             # 计算基于百分比的动态阈值
    #             threshold = np.percentile(score_map, 100 - anomaly_percent)
    #             # 二值化处理
    #             binary_map = (score_map > threshold).astype(np.uint8) * 255
    #             cv2.imwrite(save_path, binary_map)

    def visualization_seg(self, image_path_list, gt_list, pr_px, category):
        # 定义二值化阈值（0-1之间）
        BINARY_THRESHOLD = 0.5

        def normalization01(img):
            # 添加极小值防止除以0
            return (img - img.min()) / (img.max() - img.min() + 1e-10)

        if self.vis_type == 'single_norm':
            # 单图归一化模式
            for i, path in enumerate(image_path_list):
                anomaly_type = path.split('/')[-2]
                img_name = path.split('/')[-1]

                # 为所有样本生成掩码（包括正常样本）
                save_path = os.path.join(self.output_dir, category, anomaly_type)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, img_name)

                # 获取当前图像的异常得分图
                anomaly_map = pr_px[i].squeeze()

                # 归一化到0-1范围
                norm_map = normalization01(anomaly_map)

                # 二值化处理
                binary_mask = np.where(norm_map > BINARY_THRESHOLD, 255, 0).astype(np.uint8)

                # 直接保存二值图像
                cv2.imwrite(save_path, binary_mask)
    def visualization(self, image_path_list, gt_list, pr_px, category):
        def normalization01(img):
            return (img - img.min()) / (img.max() - img.min())
        if self.vis_type == 'single_norm':
            # normalized per image
            for i, path in enumerate(image_path_list):
                anomaly_type = path.split('/')[-2]
                img_name = path.split('/')[-1]
                if anomaly_type not in ['good', 'Normal', 'ok'] and gt_list[i] != 0:
                    save_path = os.path.join(self.output_dir, category, anomaly_type)
                    os.makedirs(save_path, exist_ok=True)
                    save_path = os.path.join(save_path, img_name)
                    anomaly_map = pr_px[i].squeeze()
                    anomaly_map = normalization01(anomaly_map)*255
                    anomaly_map = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)
                    cv2.imwrite(save_path, anomaly_map)
        else:
            # normalized all image
            pr_px = normalization01(pr_px)
            for i, path in enumerate(image_path_list):
                anomaly_type = path.split('/')[-2]
                img_name = path.split('/')[-1]
                save_path = os.path.join(self.output_dir, category, anomaly_type)
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, img_name)
                anomaly_map = pr_px[i].squeeze()
                anomaly_map *= 255
                anomaly_map = cv2.applyColorMap(anomaly_map.astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(save_path, anomaly_map)


    def make_category_data(self, category):
        
        print(category)
        torch.cuda.reset_max_memory_allocated()
        # divide sub-datasets
        divide_num = self.divide_num
        anomaly_maps = torch.tensor([]).double()
        anomaly_maps0 = torch.tensor([]).double()
        gt_list = []
        img_masks = []
        class_tokens1 = []
        class_tokens2 = []
        image_path_list = []
        start_time_all = time.time()

        dataset_num = 0
        # divide_iter：第几块
        for divide_iter in range(divide_num):
            #第i张图片的时候怎么处理
            test_dataset = self.load_datasets(category, divide_num=divide_num, divide_iter=divide_iter)
            #按照batch_size送入图片，怎么送入图片
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            
            # extract features
            patch_tokens_list1 = []
            patch_tokens_list2 = []
            subset_num = len(test_dataset)
            dataset_num += subset_num
            start_time = time.time()
            for image_info in tqdm(test_dataloader):
            # for image_info in test_dataloader:
                if isinstance(image_info, dict):
                    image = image_info["image"]
                    # 逐条加入
                    image_path_list.extend(image_info["image_path"])
                    img_masks.append(image_info["mask"])
                    gt_list.extend(list(image_info["is_anomaly"].numpy()))
                with torch.no_grad(), torch.cuda.amp.autocast():
                    input_image = image.to(torch.float).to(self.device) # 转成float传到gpu
                    #'dinov2'
                    #局部特征和全局特征
                    patch_tokens1 = self.dino_model.get_intermediate_layers(x=input_image, n=[l-1 for l in self.features_list], return_class_token=False)
                    image_features1 = self.dino_model(input_image)
                    # 从cpu移动到gpu
                    patch_tokens1 = [patch_tokens1[l].cpu() for l in range(len(self.features_list))]
                    fake_cls = [torch.zeros_like(p)[:, 0:1, :] for p in patch_tokens1]
                    # 补上一个0向量
                    patch_tokens1 = [torch.cat([fake_cls[i], patch_tokens1[i]], dim=1) for i in range(len(patch_tokens1))]

                     # clip
                    image_features2, patch_tokens2 = self.clip_model.encode_image(input_image, self.features_list)
                    # 归一化
                    image_features2 /= image_features2.norm(dim=-1, keepdim=True)
                    patch_tokens2 = [patch_tokens2[l].cpu() for l in range(len(self.features_list))]
                # 转成numpy图像特征
                image_features1 = [image_features1[bi].squeeze().cpu().numpy() for bi in range(image_features1.shape[0])]
                class_tokens1.extend(image_features1)
                patch_tokens_list1.append(patch_tokens1)  # (批次图片数量, patch数量L+1, 特征维度C)
                patch_tokens_list2.append(patch_tokens2)  # (批次图片数量, L+1, C)
            end_time = time.time()
            print('extract time: {}ms per image'.format((end_time-start_time)*1000/subset_num))


            #筛选评委会
            # LNAMD（固定r和layer聚合特征）
            feature_dim0 = patch_tokens_list1[0][0].shape[-1]   # 第0个batch第0层拿出最后一维：局部向量多长
            anomaly_maps_r0 = torch.tensor([]).double() 
            for r in [1]: 
                print('aggregation degree: {}'.format(r))
                #固定r
                LNAMD_r0 = LNAMD(device=self.device, r=r, feature_dim=feature_dim0, feature_layer=self.features_list)
                Z_layers0 = {}
                for im in range(len(patch_tokens_list1)):
                    #某个batch里每层的张量
                    patch_tokens0 = [p.to(self.device) for p in patch_tokens_list1[im]]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        # 聚合特征归一化,[B, L, layer, C]
                        features0 = LNAMD_r0._embed(patch_tokens0)
                        features0 /= features0.norm(dim=-1, keepdim=True)
                        # 按层缓存结果，取出一层的张量
                        for l in range(len(self.features_list)):
                            # save the aggregated features
                            if str(l) not in Z_layers0.keys():
                                Z_layers0[str(l)] = []
                                # 第 l 层、各 batch的[BLC]，batch/patch number（length）/channel
                                #'0': [(B,L,C), (B,L,C)...]
                            Z_layers0[str(l)].append(features0[:, :, l, :])
                            print("layer",l)
                            print("feature",Z_layers0[str(l)][0].shape)

                end_time = time.time()
                
                # MSM
                anomaly_maps_l0 = torch.tensor([]).double()
                start_time = time.time()
                #l = '3'
                for l in Z_layers0.keys():
                    # different layers
                    Z0 = torch.cat(Z_layers0[l], dim=0).to(self.device)  # 第 l 层列表里所有 batch 沿第 0 维拼接(N, L, C)
                    print('layer-{} mutual scoring...'.format(l))
                    anomaly_maps_msm0 = MSM(Z=Z0, device=self.device, topmin_min=0, topmin_max=0.3)
                    anomaly_maps_l0 = torch.cat((anomaly_maps_l0, anomaly_maps_msm0.unsqueeze(0).cpu()), dim=0)#沿着0拼接
                    torch.cuda.empty_cache()
                
                # 跨层数在第0层求平均
                anomaly_maps_l0 = torch.mean(anomaly_maps_l0, 0)   
                # 存到 r 累加器
                anomaly_maps_r0 = torch.cat((anomaly_maps_r0, anomaly_maps_l0.unsqueeze(0)), dim=0)
            # 对轮数r平均    
            anomaly_maps_iter0 = torch.mean(anomaly_maps_r0, 0).to(self.device)
            del anomaly_maps_r0
            torch.cuda.empty_cache()

            # interpolate
            # 【N,L】
            B0, L0 = anomaly_maps_iter0.shape
            H0 = int(np.sqrt(L0))
            # 变成原尺寸
            anomaly_maps_iter0 = F.interpolate(anomaly_maps_iter0.view(B0, 1, H0, H0),
                                              size=self.image_size, mode='bilinear', align_corners=True)
            anomaly_maps0 = torch.cat((anomaly_maps0, anomaly_maps_iter0.cpu()), dim=0)
            # 初步异常图转成numpy
            anomaly_maps0 = anomaly_maps0.cpu().numpy()
            torch.cuda.empty_cache()

            B0 = anomaly_maps0.shape[0]  # the number of unlabeled test images
            ac_score0 = np.array(anomaly_maps0).reshape(B0, -1).max(-1)
            #按比例不同取数目不同
            if len(ac_score0) <= 50:
                num_indices  = int(0.25 * len(ac_score0))
            elif 50 < len(ac_score0) <= 100:
                num_indices = int(0.20 * len(ac_score0))
            else:
                num_indices = int(0.15 * len(ac_score0))

            #sorted_indices得到排序后的索引，lowest_indices取前num_indices个索引
            sorted_indices = np.argsort(ac_score0)
            lowest_indices = sorted_indices[:num_indices]

            #real test
            anomaly_maps_r = torch.tensor([]).double()
            # LNAMD1
            feature_dim1 = patch_tokens_list1[0][0].shape[-1]
            for r in self.r_list:
                start_time = time.time()
                print('aggregation degree: {}'.format(r))
                LNAMD_r1 = LNAMD(device=self.device, r=r, feature_dim=feature_dim1, feature_layer=self.features_list)
                LNAMD_r2 = LNAMD(device=self.device, r=r, feature_dim=feature_dim1, feature_layer=self.features_list)
                Z_layers1 = {}
                Z_layers2 = {}
                #backbone1
                for im in range(len(patch_tokens_list1)):
                    patch_tokens1 = [p.to(self.device) for p in patch_tokens_list1[im]]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        features1 = LNAMD_r1._embed(patch_tokens1)
                        features1 /= features1.norm(dim=-1, keepdim=True)
                        for l in range(len(self.features_list)):
                            # save the aggregated features
                            if str(l) not in Z_layers1.keys():
                                Z_layers1[str(l)] = []
                            Z_layers1[str(l)].append(features1[:, :, l, :])
                #backbone2
                for im in range(len(patch_tokens_list2)):
                    patch_tokens2 = [p.to(self.device) for p in patch_tokens_list2[im]]
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        features2 = LNAMD_r2._embed(patch_tokens2)
                        features2 /= features2.norm(dim=-1, keepdim=True)
                        for l in range(len(self.features_list)):
                            # save the aggregated features
                            if str(l) not in Z_layers2.keys():
                                Z_layers2[str(l)] = []
                            Z_layers2[str(l)].append(features2[:, :, l, :])
                end_time = time.time()
                print('LNAMD1-{}: {}ms per image'.format(r, (end_time-start_time)*1000/subset_num))


                # MSM
                anomaly_maps_l = torch.tensor([]).double()
                start_time = time.time()
                channel_memory = ChannelMemory(max_ttl=5)
                for l in Z_layers2.keys():
                    # different layers
                    #当前特征层的局部特征
                    Z1 = torch.cat(Z_layers1[l], dim=0).to(self.device) # (N, L, C)
                    if int(l)==3 and self.use_dynamic_model:
                        for image_id in range(Z1.shape[0]):
                            current_features = Z1[image_id]
                            if image_id == 0 :
                               channel_memory.initialize(features=current_features, image_id=image_id)
                               print("initial channel",len(channel_memory.channels))
                            else:
                                channel_memory.update(features=current_features, image_id=image_id)
                                print("update channel",len(channel_memory.channels))

                         

                    #从 Z1 中取出“可信正常图片”的 DINO 局部特征
                    Z11 = Z1[lowest_indices]

                    Z2 = torch.cat(Z_layers2[l], dim=0).to(self.device)  # (N, L, C)
                    Z22 = Z2[lowest_indices]
            

                    train_samples = []
                    image_num, patch_num, c = Z11.shape
                    for x in range(Z11.shape[0]):
                        train_sample1 = torch.cdist(Z11[x:x + 1], Z11.reshape(-1, c)).reshape(patch_num, -1, patch_num)
                        train_sample1 = torch.min(train_sample1, -1)[0]
                        train_sample1 = torch.flatten(train_sample1)
                        train_sample1 = train_sample1.unsqueeze(0)
                        train_sample2 = torch.cdist(Z22[x:x + 1], Z22.reshape(-1, c)).reshape(patch_num, -1, patch_num)
                        train_sample2 = torch.min(train_sample2, -1)[0]
                        train_sample2 = torch.flatten(train_sample2)
                        train_sample2 = train_sample2.unsqueeze(0)
                        train_sample = torch.cat((train_sample1, train_sample2*0.5), dim=0).T
                        train_samples.append(train_sample)
                    train_data = torch.cat(train_samples, dim=0)
                    train_data = train_data.cpu()
                    self.detect_fuser.fit(train_data)
                    print('layer-{} mutual scoring...'.format(l))
                    anomaly_maps_msm = MSM2(Z=Z1,Z11=Z11,Z2=Z2,Z22=Z22,detect_fuser=self.detect_fuser, device=self.device, topmin_min=0.02, topmin_max=0.3)
                    anomaly_maps_l = torch.cat((anomaly_maps_l, anomaly_maps_msm.unsqueeze(0).cpu()), dim=0)
                    torch.cuda.empty_cache()
                anomaly_maps_l = torch.mean(anomaly_maps_l, 0)
                anomaly_maps_r = torch.cat((anomaly_maps_r, anomaly_maps_l.unsqueeze(0)), dim=0)
                end_time = time.time()
                print('MSM: {}ms per image'.format((end_time-start_time)*1000/subset_num))
            anomaly_maps_iter = torch.mean(anomaly_maps_r, 0).to(self.device)
            del anomaly_maps_r
            torch.cuda.empty_cache()

            # interpolate
            B, L = anomaly_maps_iter.shape
            H = int(np.sqrt(L))
            anomaly_maps_iter = F.interpolate(anomaly_maps_iter.view(B, 1, H, H),
                                        size=self.image_size, mode='bilinear', align_corners=True)
            anomaly_maps = torch.cat((anomaly_maps, anomaly_maps_iter.cpu()), dim=0)

        end_time_all = time.time()
        print('MuSc: {}ms per image'.format((end_time_all-start_time_all)*1000/dataset_num))
        print(f"最大GPU内存使用: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")
        anomaly_maps = anomaly_maps.cpu().numpy()
        torch.cuda.empty_cache()

        B = anomaly_maps.shape[0]   # the number of unlabeled test images
        ac_score = np.array(anomaly_maps).reshape(B, -1).max(-1)
        # RsCIN
        if self.dataset == 'visa':
            k_score = [1, 8, 9]
        elif self.dataset == 'mvtec_ad':
            k_score = [1, 2, 3]
        else:
            k_score = [1, 2, 3]
                                                        # 之前保存的每张图片的 DINO 整图特征
        scores_cls = RsCIN(ac_score, class_tokens1, k_list=k_score)

        print('computing metrics...')
        # 每张图真实是否异常的标签
        pr_sp = np.array(scores_cls)
        # 每张图预测异常分数
        gt_sp = np.array(gt_list)
        gt_px = torch.cat(img_masks, dim=0).numpy().astype(np.int32)
        pr_px = np.array(anomaly_maps)
        image_metric, pixel_metric = compute_metrics(gt_sp, pr_sp, gt_px, pr_px)
        auroc_sp, f1_sp, ap_sp = image_metric
        auroc_px, f1_px, ap_px, aupro = pixel_metric
        print(category)
        print('image-level, auroc:{}, f1:{}, ap:{}'.format(auroc_sp*100, f1_sp*100, ap_sp*100))
        print('pixel-level, auroc:{}, f1:{}, ap:{}, aupro:{}'.format(auroc_px*100, f1_px*100, ap_px*100, aupro*100))

        if self.vis:
            print('visualization...')
            self.visualization_seg(image_path_list, gt_list, pr_px, category)
    
        return image_metric, pixel_metric


    def main(self):
        auroc_sp_ls = []
        f1_sp_ls = []
        ap_sp_ls = []
        auroc_px_ls = []
        f1_px_ls = []
        ap_px_ls = []
        aupro_ls = []
        for category in self.categories:
            image_metric, pixel_metric = self.make_category_data(category=category,)
            auroc_sp, f1_sp, ap_sp = image_metric
            auroc_px, f1_px, ap_px, aupro = pixel_metric
            auroc_sp_ls.append(auroc_sp)
            f1_sp_ls.append(f1_sp)
            ap_sp_ls.append(ap_sp)
            auroc_px_ls.append(auroc_px)
            f1_px_ls.append(f1_px)
            ap_px_ls.append(ap_px)
            aupro_ls.append(aupro)
        # mean算术平均
        auroc_sp_mean = sum(auroc_sp_ls) / len(auroc_sp_ls)
        f1_sp_mean = sum(f1_sp_ls) / len(f1_sp_ls)
        ap_sp_mean = sum(ap_sp_ls) / len(ap_sp_ls)
        auroc_px_mean = sum(auroc_px_ls) / len(auroc_px_ls)
        f1_px_mean = sum(f1_px_ls) / len(f1_px_ls)
        ap_px_mean = sum(ap_px_ls) / len(ap_px_ls)
        aupro_mean = sum(aupro_ls) / len(aupro_ls)

        for i, category in enumerate(self.categories):
            print(category)
            print('image-level, auroc:{}, f1:{}, ap:{}'.format(auroc_sp_ls[i]*100, f1_sp_ls[i]*100, ap_sp_ls[i]*100))
            print('pixel-level, auroc:{}, f1:{}, ap:{}, aupro:{}'.format(auroc_px_ls[i]*100, f1_px_ls[i]*100, ap_px_ls[i]*100, aupro_ls[i]*100))
        print('mean')
        print('image-level, auroc:{}, f1:{}, ap:{}'.format(auroc_sp_mean*100, f1_sp_mean*100, ap_sp_mean*100))
        print('pixel-level, auroc:{}, f1:{}, ap:{}, aupro:{}'.format(auroc_px_mean*100, f1_px_mean*100, ap_px_mean*100, aupro_mean*100))
        
        # save in excel
        if self.save_excel:
            # 激活工作区
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "MuSc_results"
            sheet.cell(row=1,column=2,value='auroc_px')
            sheet.cell(row=1,column=3,value='f1_px')
            sheet.cell(row=1,column=4,value='ap_px')
            sheet.cell(row=1,column=5,value='aupro')
            sheet.cell(row=1,column=6,value='auroc_sp')
            sheet.cell(row=1,column=7,value='f1_sp')
            sheet.cell(row=1,column=8,value='ap_sp')
            for col_index in range(2):
                for row_index in range(len(self.categories)):
                    if col_index == 0:
                        sheet.cell(row=row_index+2,column=col_index+1,value=self.categories[row_index])
                    else:
                        sheet.cell(row=row_index+2,column=col_index+1,value=auroc_px_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+2,value=f1_px_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+3,value=ap_px_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+4,value=aupro_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+5,value=auroc_sp_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+6,value=f1_sp_ls[row_index]*100)
                        sheet.cell(row=row_index+2,column=col_index+7,value=ap_sp_ls[row_index]*100)
                    if row_index == len(self.categories)-1:
                        if col_index == 0:
                            sheet.cell(row=row_index+3,column=col_index+1,value='mean')
                        else:
                            sheet.cell(row=row_index+3,column=col_index+1,value=auroc_px_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+2,value=f1_px_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+3,value=ap_px_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+4,value=aupro_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+5,value=auroc_sp_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+6,value=f1_sp_mean*100)
                            sheet.cell(row=row_index+3,column=col_index+7,value=ap_sp_mean*100)
            workbook.save(os.path.join(self.output_dir, 'results.xlsx'))


