# 简介
本模块用于实现 Mask-RCNN 和 Sparse-RCNN 的目标检测与图像分割任务。

# VOC 数据集
PASCAL VOC（Pattern Analysis, Statistical Modelling and Computational Learning Visual Object Classes）数据集是计算机视觉领域中最具代表性和影响力的标准数据集之一，广泛用于目标检测、语义分割、实例分割等任务的研究。VOC 数据集最初由欧洲计算机视觉研究机构（如剑桥大学）在 2005 年推出，目标是推动计算机视觉领域在多个任务上的研究和发展。

VOC 数据集的主要特点包括：

1. 多任务：支持目标检测、语义分割、实例分割、动作识别等多种视觉任务。
2. 丰富的标注信息：包含图片中的物体类别、边界框（bounding box）、物体掩码（mask）等多种标注信息。
3. 数据规模适中：数据集包含多达20个类别的物体（如猫、狗、鸟、车等），并且具有大约11,000张图像，适合进行大规模的模型训练和评估。

VOC 没有提供自带的 mask 掩码，将其修改为了 COCO 相同格式的 mask。

# 环境配置
- 建议使用conda新建环境（如mmdet），Python推荐3.8/3.9。

- 安装PyTorch（版本需与CUDA匹配），如：
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

- 安装MMCV（需与PyTorch和CUDA版本对应）
```
pip insatll "mmcv>=2.1.0"
```
- 安装本项目依赖
```
pip install -r requirements.txt
```


# Mask-RCNN 训练与测试
- 训练
多卡：
```
bash tools/dist_train.sh configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc0712.py 8
```
- 测试
```
python tools/test.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc0712.py work_dirs/mask-rcnn_r50_fpn_1x_voc0712/latest.pth --show-dir results/maskrcnn
```


# Sparse-RCNN 训练与测试
- 训练
```
bash tools/dist_train.sh configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc0712.py 8
```
- 测试
```
python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc0712.py work_dirs/sparse-rcnn_r50_fpn_1x_voc0712/latest.pth --show-dir results/sparsercnn
```
