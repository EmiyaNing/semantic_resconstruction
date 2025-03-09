# python 内置库最先import
import os
import argparse

# 放在第二的是 import整个包的代码
import torch
import json

# 放在第三的是 使用from关键字从第三方包中导入某个function或者class的代码
from PIL import Image

# 放在最后import的代码是本代码库中实现的代码
from horizon_tools import *
from mmdet3d.registry import MODELS
from mmdet3d.testing import get_model_cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('--config', default="_base_/models/horizonnet.py",help='train config file path')
    parser.add_argument('--work_dir',help='the dir to save logs and models')
    parser.add_argument('--ckpt', default='./tools/pretrain/resnet50_rnn__mp3d.pth')
    parser.add_argument('--img_path', default='./tools/assets/conferenceRoom_1_rgb.png')
    parser.add_argument('--output_dir', default='./tools/assets/inferenced')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    horizon_cfg = get_model_cfg(args.config)
    horizonnet  = MODELS.build(horizon_cfg)

    ckpt        = torch.load(args.ckpt)
    horizonnet.load_state_dict(ckpt)
    horizonnet.cuda()

    k           = os.path.split(args.img_path)[-1][:-4]
    img_pil     = Image.open(args.img_path)
    if img_pil.size != (1024, 512):
        img_pil = img_pil.resize((1024, 512), Image.BICUBIC)

    img_ori     = np.array(img_pil)[..., :3].transpose([2, 0, 1]).copy()
    img_tensor  = torch.tensor([img_ori / 255],dtype=torch.float32)

    #import pdb
    #pdb.set_trace()
    cor_id, z0, z1, vis_out = inference(
        net=horizonnet,
        x = img_tensor,
        device='cuda',
        visualize=True,
        r=0.05
    )

    with open(os.path.join(args.output_dir, k + '.json'), 'w') as f:
        json.dump(
            {
                'z0': float(z0),
                'z1': float(z1),
                'uv': [[float(u), float(v)] for u, v in cor_id],
            }, f)
        
    if vis_out is not None:
        vis_path = os.path.join(args.output_dir, k + '_raw.png')
        vh, vw   = vis_out.shape[:2]
        Image.fromarray(vis_out) \
             .resize((vw // 2, vh // 2), Image.LANCZOS) \
             .save(vis_path)