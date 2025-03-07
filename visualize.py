import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import os
import sys
import glob
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import HorizonNet
from dataset import visualize_a_data
from misc import post_proc, panostretch, utils
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
def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def augment(x_img, flip, rotate):
    x_img = x_img.numpy()
    aug_type = ['']
    x_imgs_augmented = [x_img]
    if flip:
        aug_type.append('flip')
        x_imgs_augmented.append(np.flip(x_img, axis=-1))
    for shift_p in rotate:
        shift = int(round(shift_p * x_img.shape[-1]))
        aug_type.append('rotate %d' % shift)
        x_imgs_augmented.append(np.roll(x_img, shift, axis=-1))
    return torch.FloatTensor(np.concatenate(x_imgs_augmented, 0)), aug_type


def augment_undo(x_imgs_augmented, aug_type):
    x_imgs_augmented = x_imgs_augmented.cpu().numpy()
    sz = x_imgs_augmented.shape[0] // len(aug_type)
    x_imgs = []
    for i, aug in enumerate(aug_type):
        x_img = x_imgs_augmented[i*sz : (i+1)*sz]
        if aug == 'flip':
            x_imgs.append(np.flip(x_img, axis=-1))
        elif aug.startswith('rotate'):
            shift = int(aug.split()[-1])
            x_imgs.append(np.roll(x_img, -shift, axis=-1))
        elif aug == '':
            x_imgs.append(x_img)
        else:
            raise NotImplementedError()

    return np.array(x_imgs)


def inference(net, x, device, flip=False, rotate=[], visualize=False,
              force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''

    H, W = tuple(x.shape[2:])

    # Network feedforward (with testing augmentation)
    x, aug_type = augment(x, flip, rotate)
    y_bon_, y_cor_ = net(x.to(device))
    y_bon_ = augment_undo(y_bon_.cpu(), aug_type).mean(0)
    y_cor_ = augment_undo(torch.sigmoid(y_cor_).cpu(), aug_type).mean(0)

    # Visualize raw model output
    if visualize:
        vis_out = visualize_a_data(x[0],
                                   torch.FloatTensor(y_bon_[0]),
                                   torch.FloatTensor(y_cor_[0]))
    else:
        vis_out = None

    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0, 0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    if force_raw:
        # Do not run post-processing, export raw polygon (1024*2 vertices) instead.
        # [TODO] Current post-processing lead to bad results on complex layout.
        cor = np.stack([np.arange(1024), y_bon_[0]], 1)

    else:
        # Detech wall-wall peaks
        if min_v is None:
            min_v = 0 if force_cuboid else 0.05
        r = int(round(W * r / 2))
        N = 4 if force_cuboid else None
        xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

        # Generate wall-walls
        cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
        if not force_cuboid:
            # Check valid (for fear self-intersection)
            xy2d = np.zeros((len(xy_cor), 2), np.float32)
            for i in range(len(xy_cor)):
                xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
            if not Polygon(xy2d).is_valid:
                print(
                    'Fail to generate valid general layout!! '
                    'Generate cuboid as fallback.',
                    file=sys.stderr)
                xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
                cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # Normalized to [0, 1]
    cor_id[:, 0] /= W
    cor_id[:, 1] /= H

    return cor_id, z0, z1, vis_out

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
check_point="E:\mmdetection3d\\tools\work_dirs\\resnet50_rnn__mp3d.pth"
ckpt = torch.load(check_point, map_location='cpu')
horizonnet=runner.model
runner.load_or_resume()
state_dict=horizonnet.state_dict()
#E:\HorizonNet-master\\resnet50_rnn__mp3d.pth
img_path="E:\mmdetection3d\\assets\conferenceRoom_1_rgb_aligned_rgb.png"
device = 'cuda'
output_dir="E:/mmdetection3d/assets/inferenced"
if not os.path.isdir(output_dir):
    print('Output directory %s not existed. Create one.' % output_dir)
    os.makedirs(args.output_dir)

with torch.no_grad():
        k = os.path.split(img_path)[-1][:-4]
        # Load image
        img_pil = Image.open(img_path)
        if img_pil.size != (1024, 512):
            img_pil = img_pil.resize((1024, 512), Image.BICUBIC)
        img_ori = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
        x = torch.FloatTensor([img_ori / 255])
        input_dict={}
        input_dict["imgs"]=x
        # Inferenceing corners
        cor_id, z0, z1, vis_out = inference(net=horizonnet, x=x, device=device,visualize=True,r=0.05)
        # Output result
        with open(os.path.join(output_dir, k + '.json'), 'w') as f:
            json.dump({
                'z0': float(z0),
                'z1': float(z1),
                'uv': [[float(u), float(v)] for u, v in cor_id],
            }, f)

        if vis_out is not None:
            vis_path = os.path.join(output_dir, k + '.raw.png')
            vh, vw = vis_out.shape[:2]
            Image.fromarray(vis_out) \
                .resize((vw // 2, vh // 2), Image.LANCZOS) \
                .save(vis_path)