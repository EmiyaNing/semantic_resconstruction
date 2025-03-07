import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend
import re
from collections import OrderedDict
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', default="E:/mmdetection3d/configs/pointnet2/pointnet2_msg_panorama_test.py",help='train config file path')
    parser.add_argument('--work_dir',help='the dir to save logs and models')
    # print(parser.config)
    # input()
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--sync_bn',
        choices=['none', 'torch', 'mmcv'],
        default='none',
        help='convert all BatchNorm layers in the model to SyncBatchNorm '
        '(SyncBN) or mmcv.ops.sync_bn.SyncBatchNorm (MMSyncBN) layers.')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        default="E:/mmdetection3d/tools/work_dirs/resnet50_rnn__mp3d.pth",
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
args = parse_args()
# print(args.config)
# load config
cfg = Config.fromfile(args.config)
# print(cfg)
# TODO: We will unify the ceph support approach with other OpenMMLab repos
if args.ceph:
    cfg = replace_ceph_backend(cfg)
cfg.launcher = args.launcher
if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)
# work_dir is determined in this priority: CLI > segment in file > filename
if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
    # use config filename as default work_dir if cfg.work_dir is None
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    # print(cfg.work_dir)
# enable automatic-mixed-precision training
if args.amp is True:
    optim_wrapper = cfg.optim_wrapper.type
    if optim_wrapper == 'AmpOptimWrapper':
        print_log(
            'AMP training is already enabled in your config.',
            logger='current',
            level=logging.WARNING)
    else:
        assert optim_wrapper == 'OptimWrapper', (
            '`--amp` is only supported when the optimizer wrapper type is '
            f'`OptimWrapper` but got {optim_wrapper}.')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'
# convert BatchNorm layers
if args.sync_bn != 'none':
    cfg.sync_bn = args.sync_bn
# enable automatically scaling LR
if args.auto_scale_lr:
    if 'auto_scale_lr' in cfg and \
            'enable' in cfg.auto_scale_lr and \
            'base_batch_size' in cfg.auto_scale_lr:
        cfg.auto_scale_lr.enable = True
    else:
        raise RuntimeError('Can not find "auto_scale_lr" or '
                           '"auto_scale_lr.enable" or '
                           '"auto_scale_lr.base_batch_size" in your'
                           ' configuration file.')

# resume is determined in this priority: resume from > auto_resume
if args.resume == 'auto':
    cfg.resume = None
    cfg.load_from = args.resume
elif args.resume is not None:
    cfg.resume = False
    cfg.load_from = args.resume

# build the runner from config
if 'runner_type' not in cfg:
    # build the default runner
    runner = Runner.from_cfg(cfg)
else:
    # build customized runner from the registry
    # if 'runner_type' is set in the cfg
    runner = RUNNERS.build(cfg)
check_point="E:\HorizonNet-master\\resnet50_rnn__mp3d.pth"
ckpt = torch.load(check_point, map_location='cpu')
horizonnet=runner.model
super_ckpt=horizonnet.state_dict()
#E:\HorizonNet-master\\resnet50_rnn__mp3d.pth
if 'state_dict' in ckpt:
    _state_dict = ckpt['state_dict']
elif 'model' in ckpt:
    _state_dict = ckpt['model']
else:
    _state_dict = ckpt

new_state_dict=OrderedDict()
# 定义修订规则
for key,value in _state_dict.items():
    key="backbone."+key
    key=key.replace(".encoder","")
    new_state_dict[key]=value
torch.save(new_state_dict, "E:/mmdetection3d/tools/work_dirs/resnet50_rnn__mp3d.pth")

print("字典已保存为 my_dict.pth 文件")

# 修改键名



# 打印修改后的 state_dict


# state_dict = OrderedDict()
# for k, v in _state_dict.items():
#     if k.startswith('backbone.'):
#         state_dict[k[9:]] = v
#
# # strip prefix of state_dict
# if list(state_dict.keys())[0].startswith('module.'):
#     state_dict = {k[7:]: v for k, v in state_dict.items()}
#
# # reshape absolute position embedding
# if state_dict.get('absolute_pos_embed') is not None:
#     absolute_pos_embed = state_dict['absolute_pos_embed']
#     N1, L, C1 = absolute_pos_embed.size()
#     N2, C2, H, W = self.absolute_pos_embed.size()
#     if N1 != N2 or C1 != C2 or L != H * W:
#         print('Error in loading absolute_pos_embed, pass')
#     else:
#         state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
#             N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
#
# # interpolate position bias table if needed
# relative_position_bias_table_keys = [
#     k for k in state_dict.keys()
#     if 'relative_position_bias_table' in k
# ]
# for table_key in relative_position_bias_table_keys:
#     table_pretrained = state_dict[table_key]
#     table_current = self.state_dict()[table_key]
#     L1, nH1 = table_pretrained.size()
#     L2, nH2 = table_current.size()
#     if nH1 != nH2:
#         print(f'Error in loading {table_key}, pass')
#     elif L1 != L2:
#         S1 = int(L1 ** 0.5)
#         S2 = int(L2 ** 0.5)
#         table_pretrained_resized = F.interpolate(
#             table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
#             size=(S2, S2),
#             mode='bicubic')
#         state_dict[table_key] = table_pretrained_resized.view(
#             nH2, L2).permute(1, 0).contiguous()
# def swin_converter(ckpt):
#
#     new_ckpt = OrderedDict()
#
#     def correct_unfold_reduction_order(x):
#         out_channel, in_channel = x.shape
#         x = x.reshape(out_channel, 4, in_channel // 4)
#         x = x[:, [0, 2, 1, 3], :].transpose(1,
#                                             2).reshape(out_channel, in_channel)
#         return x
#
#     def correct_unfold_norm_order(x):
#         in_channel = x.shape[0]
#         x = x.reshape(4, in_channel // 4)
#         x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
#         return x
#
#     for k, v in ckpt.items():
#         if k.startswith('head'):
#             continue
#         elif k.startswith('layers'):
#             new_v = v
#             if 'attn.' in k:
#                 new_k = k.replace('attn.', 'attn.w_msa.')
#             elif 'mlp.' in k:
#                 if 'mlp.fc1.' in k:
#                     new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
#                 elif 'mlp.fc2.' in k:
#                     new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
#                 else:
#                     new_k = k.replace('mlp.', 'ffn.')
#             elif 'downsample' in k:
#                 new_k = k
#                 if 'reduction.' in k:
#                     new_v = correct_unfold_reduction_order(v)
#                 elif 'norm.' in k:
#                     new_v = correct_unfold_norm_order(v)
#             else:
#                 new_k = k
#             new_k = new_k.replace('layers', 'stages', 1)
#         elif k.startswith('patch_embed'):
#             new_v = v
#             if 'proj' in k:
#                 new_k = k.replace('proj', 'projection')
#             else:
#                 new_k = k
#         else:
#             new_v = v
#             new_k = k
#
#         new_ckpt['backbone.' + new_k] = new_v
#
#     return new_ckpt
# load state_dict


