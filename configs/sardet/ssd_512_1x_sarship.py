# model settings
input_size = 512
num_classes = 1
model = dict(
    type='SingleStageDetector',
    # pretrained='open-mmlab://vgg16_caffe',
    pretrained=None,
    backbone=dict(
        type='SSDVGG',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        num_classes=num_classes,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.1, 0.9),
            strides=[8, 16, 32, 64, 128, 256, 512],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_threshold=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
# dataset setting
dataset_type = 'CocoDataset'
classes = ('ship',)
data_root = 'data/SSDD/SSDD_coco/'
img_norm_cfg = dict(
    mean=[98.13131, 98.13131, 98.13131], std=[1.0, 1.0, 1.0], to_rgb=True)
train_scale = 512
train_pipeline = [
    # dict(type='LoadTiffImageFromFile', to_float32=True),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(train_scale, train_scale), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_scale = 512
test_pipeline = [
    # dict(type='LoadTiffImageFromFile', to_float32=True),
    dict(type='LoadImageFromFile', to_float32=True),
    # dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(test_scale, test_scale),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            dict(type='Collect', keys=['img']),
        ])
]
batch_per_gpu = 8
data = dict(
    samples_per_gpu=batch_per_gpu,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
                type=dataset_type,
                classes=classes,
                ann_file=data_root + 'annotations/instances_sarship_train.json',
                img_prefix=data_root + 'train/',
                pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_sarship_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations/instances_sarship_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))
evaluation = dict(interval=10, metric='bbox')
# optimizer
# lr = 2e-3  # 0.955 300
lr = 1e-3
# lr = 1e-3 / 2  # 0.936 280
total_epochs = 300
optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[total_epochs * 2 // 3, total_epochs * 8 // 9])
checkpoint_config = dict(interval=5)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]