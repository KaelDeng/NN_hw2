# import os
# import xml.etree.ElementTree as ET
# from PIL import Image
# import numpy as np


# voc_root = 'data/VOCdevkit'
# years = ['VOC2007', 'VOC2012']
# for year in years:
#     seg_dir = os.path.join(voc_root, year, 'SegmentationObject')
#     ann_dir = os.path.join(voc_root, year, 'Annotations')
#     imgset_file = os.path.join(voc_root, year, 'ImageSets/Segmentation/trainval.txt')
#     with open(imgset_file) as f:
#         img_ids = [x.strip() for x in f.readlines()]
#     valid_imgs = []
#     for img_id in img_ids:
#         mask_path = os.path.join(seg_dir, img_id + '.png')
#         ann_path = os.path.join(ann_dir, img_id + '.xml')
#         if not os.path.exists(mask_path) or not os.path.exists(ann_path):
#             continue
#         mask = np.array(Image.open(mask_path))
#         tree = ET.parse(ann_path)
#         root = tree.getroot()
#         valid = True
#         for obj in root.findall('object'):
#             if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
#                 continue
#             bndbox = obj.find('bndbox')
#             xmin = int(float(bndbox.find('xmin').text)) - 1
#             ymin = int(float(bndbox.find('ymin').text)) - 1
#             xmax = int(float(bndbox.find('xmax').text)) - 1
#             ymax = int(float(bndbox.find('ymax').text)) - 1
#             # 检查bbox区域内是否有非0像素（即有mask）
#             if np.sum(mask[ymin:ymax+1, xmin:xmax+1] > 0) == 0:
#                 valid = False
#                 break
#         if valid:
#             valid_imgs.append(img_id)
#     # 保存新列表
#     with open(os.path.join(voc_root, year, 'ImageSets/Segmentation/trainval_with_full_mask.txt'), 'w') as f:
#         for img_id in valid_imgs:
#             f.write(img_id + '\n')
#     print(f'{year} 有完整掩码的图片数: {len(valid_imgs)}')


import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

# voc_root = r'D:\Course Work\Neural Network\mmdetection-main\mmdetection-main\data\VOCdevkit'
# year = 'VOC2007'
# seg_dir = os.path.join(voc_root, year, 'SegmentationObject')
# ann_dir = os.path.join(voc_root, year, 'Annotations')
# imgset_file = os.path.join(voc_root, year, 'ImageSets/Segmentation/trainval.txt')
# with open(imgset_file) as f:
#     img_ids = [x.strip() for x in f.readlines()]
# valid_imgs = []
# invalid_imgs = []
# for img_id in img_ids:
#     mask_path = os.path.join(seg_dir, img_id + '.png')
#     ann_path = os.path.join(ann_dir, img_id + '.xml')
#     if not os.path.exists(mask_path) or not os.path.exists(ann_path):
#         continue
#     mask = np.array(Image.open(mask_path))
#     tree = ET.parse(ann_path)
#     root = tree.getroot()
#     valid = True
#     for obj in root.findall('object'):
#         if obj.find('difficult') is not None and int(obj.find('difficult').text) == 1:
#             continue
#         bndbox = obj.find('bndbox')
#         xmin = int(float(bndbox.find('xmin').text)) - 1
#         ymin = int(float(bndbox.find('ymin').text)) - 1
#         xmax = int(float(bndbox.find('xmax').text)) - 1
#         ymax = int(float(bndbox.find('ymax').text)) - 1
#         # 边界检查
#         xmin = max(xmin, 0)
#         ymin = max(ymin, 0)
#         xmax = min(xmax, mask.shape[1] - 1)
#         ymax = min(ymax, mask.shape[0] - 1)
#         if np.sum(mask[ymin:ymax+1, xmin:xmax+1] > 0) == 0:
#             valid = False
#             print(f'图片{img_id}的目标没有mask: bbox=({xmin},{ymin},{xmax},{ymax})')
#     if valid:
#         valid_imgs.append(img_id)
#     else:
#         invalid_imgs.append(img_id)
# print(f'有完整掩码的图片数: {len(valid_imgs)}')
# print(f'有掩码文件但不完整的图片数: {len(invalid_imgs)}')

# seg_dir = r'D:\Course Work\Neural Network\mmdetection-main\mmdetection-main\data\VOCdevkit\VOC2007\SegmentationObject'
# seg_imgs = set([f.split('.')[0] for f in os.listdir(seg_dir) if f.endswith('.png')])
# with open('data/VOCdevkit/VOC2007/ImageSets/Segmentation/trainval_with_full_mask.txt') as f:
#     txt_imgs = set([x.strip() for x in f.readlines()])
# print('txt中有但掩码文件夹没有的图片:', txt_imgs - seg_imgs)
# print('掩码文件夹有但txt没有的图片:', seg_imgs - txt_imgs)


import mmdet.datasets.voc  # 强制注册
from mmengine.config import Config
from mmdet.registry import DATASETS

cfg = Config.fromfile(r'configs\_base_\datasets\voc0712.py')
dataset_cfg = cfg.train_dataloader.dataset
print(DATASETS.module_dict.keys())  # 打印注册表内容
dataset = DATASETS.build(dataset_cfg)
for i in range(len(dataset)):
    data = dataset[i]
    if 'masks' not in data['instances']:
        print(f'第{i}张图片没有masks字段，图片名：{data["img_path"]}')