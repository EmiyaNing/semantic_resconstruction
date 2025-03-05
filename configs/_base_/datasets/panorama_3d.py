# dataset settings
dataset_type = 'PanoramaDataset'
data_root = '/autodl-fs/data/mmdetection3d/data/s3dis/'
backend_args = None

metainfo = dict(classes=('table', 'chair', 'sofa', 'bookcase', 'board'))
train_area = [1]
test_area = 1

train_pipeline = [
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='RandomResize', scale=[(512, 384), (768, 576)], keep_ratio=True),
    dict(
        type='Pack3DDetInputs',keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    # dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='Pack3DDetInputs', keys=['img'])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=13,
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file=f'panorama_infos_Area_{i}.pkl',
                    pipeline=train_pipeline,
                    filter_empty_gt=True,
                    metainfo=metainfo,
                    box_type_3d='Depth',
                    backend_args=backend_args) for i in train_area
            ])))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f'panorama_infos_Area_{test_area}.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        test_mode=True,
        box_type_3d='Depth',
        backend_args=backend_args))
test_dataloader = val_dataloader
val_evaluator = dict(type='IndoorMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
