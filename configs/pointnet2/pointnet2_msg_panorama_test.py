_base_ = [
    '../_base_/datasets/panorama_3d.py', '../_base_/models/horizonnet.py',
    '../_base_/schedules/seg-cosine-50e.py', '../_base_/default_runtime.py'
]
#这里进行修改过的
# model settings


# data settings
train_dataloader = dict(batch_size=16)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2))

# PointNet2-MSG needs longer training time than PointNet2-SSG
train_cfg = dict(by_epoch=True, max_epochs=80, val_interval=2)
# customhook是需要考虑要不要的