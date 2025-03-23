model = dict(
    type='New3DDetector',
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    backbone=dict(
        type="Panoformer",
    ),
    #这里的out_channels不知道是多少
    neck=dict(
        type='TR3DNeck', in_channels=(256, 512, 1024, 2048), out_channels=128),
    bbox_head=dict(
        type='TR3DHead',
        in_channels=128,
        voxel_size=0.01,
        pts_center_threshold=6,
        num_reg_outs=6),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.0001),
    clip_grad=dict(max_norm=10, norm_type=2))

# learning rate
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=12,
    by_epoch=True,
    milestones=[8, 11],
    gamma=0.1)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')