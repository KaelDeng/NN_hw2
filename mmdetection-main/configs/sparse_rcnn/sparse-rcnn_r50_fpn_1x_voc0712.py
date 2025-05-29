# VOC 20类 Sparse R-CNN 完整配置

_base_ = [
    '../_base_/models/sparse-rcnn_r50_fpn.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 数据集根目录
data_root = 'data/VOCdevkit/'

# 修改类别数为20（VOC）
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20)
    )
)

# 预训练权重（COCO）
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20210223_214621-0c1dfdb1.pth'

# 训练数据加载器
train_dataloader = dict(
    batch_size=2,  # 单卡2，8卡可设16
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type='VOCDataset',
                    data_root=data_root,
                    ann_file='VOC2007/ImageSets/Segmentation/trainval_with_full_mask.txt',
                    data_prefix=dict(sub_data_root='VOC2007/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                        dict(type='Resize', scale=(1000, 600), keep_ratio=True),
                        dict(type='RandomFlip', prob=0.5),
                        dict(type='PackDetInputs'),
                    ]),
                dict(
                    type='VOCDataset',
                    data_root=data_root,
                    ann_file='VOC2012/ImageSets/Segmentation/trainval_with_full_mask.txt',
                    data_prefix=dict(sub_data_root='VOC2012/'),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=[
                        dict(type='LoadImageFromFile'),
                        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                        dict(type='Resize', scale=(1000, 600), keep_ratio=True),
                        dict(type='RandomFlip', prob=0.5),
                        dict(type='PackDetInputs'),
                    ]),
            ]
        )
    )
)

# 验证与测试数据加载器
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VOCDataset',
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Segmentation/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
            ),
        ]
    )
)
test_dataloader = val_dataloader

# 评测方式
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

# 优化器与学习率
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
)

# 训练轮数
max_epochs = 12
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

# 其余参数沿用 _base_ 配置

