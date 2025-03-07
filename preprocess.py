'''
This script preprocess the given 360 panorama image under euqirectangular projection
and dump them to the given directory for further layout prediction and visualization.
The script will:
    - extract and dump the vanishing points
    - rotate the equirect image to align with the detected VP
    - extract the VP aligned line segments (for further layout prediction model)
The dump files:
    - `*_VP.txt` is the vanishg points
    - `*_aligned_rgb.png` is the VP aligned RGB image
    - `*_aligned_line.png` is the VP aligned line segments images

Author: Cheng Sun
Email : chengsun@gapp.nthu.edu.tw
'''

import os
import glob
import argparse
import numpy as np
from PIL import Image

from misc.pano_lsd_align import panoEdgeDetection, rotatePanorama


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
# I/O related arguments
parser.add_argument('--img_glob',default="E:\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1\conferenceRoom_2\camera_e769075d0f9f4b8ca9baedc5c0a20799_conferenceRoom_2_frame_equirectangular_domain_rgb.png",
                    help='NOTE: Remeber to quote your glob path.')
parser.add_argument('--output_dir',default="E:\mmdetection3d\\assets")
parser.add_argument('--rgbonly', default=False,action='store_true',
                    help='Add this if use are preparing customer dataset')
# Preprocessing related arguments
parser.add_argument('--q_error', default=0.7, type=float)
parser.add_argument('--refine_iter', default=3, type=int)
args = parser.parse_args()

paths = sorted(glob.glob(args.img_glob))
if len(paths) == 0:
    print('no images found')

# Check given path exist
for path in paths:
    assert os.path.isfile(path), '%s not found' % path

# Check target directory
if not os.path.isdir(args.output_dir):
    print('Output directory %s not existed. Create one.')
    os.makedirs(args.output_dir)

# Process each input
for i_path in paths:
    print('Processing', i_path, flush=True)

    # Load and cat input images
    img_ori = np.array(Image.open(i_path).resize((1024, 512), Image.BICUBIC))[..., :3]
    # print(np.array(Image.open(i_path).resize((1024,512),Image.BICUBIC))[...,:3])
    # VP detection and line segment extraction
    _, vp, _, _, panoEdge, _, _ = panoEdgeDetection(img_ori,
                                                    qError=args.q_error,
                                                    refineIter=args.refine_iter)
    panoEdge = (panoEdge > 0)

    # Align images with VP
    i_img = rotatePanorama(img_ori / 255.0, vp[2::-1])
    l_img = rotatePanorama(panoEdge.astype(np.float32), vp[2::-1])

    # Dump results
    basename = os.path.splitext(os.path.basename(i_path))[0]
    if args.rgbonly:
        path = os.path.join(args.output_dir, '%s.png' % basename)
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path)
    else:
        path_VP = os.path.join(args.output_dir, '%s_VP.txt' % basename)
        path_i_img = os.path.join(args.output_dir, '%s_aligned_rgb.png' % basename)
        path_l_img = os.path.join(args.output_dir, '%s_aligned_line.png' % basename)

        with open(path_VP, 'w') as f:
            for i in range(3):
                f.write('%.6f %.6f %.6f\n' % (vp[i, 0], vp[i, 1], vp[i, 2]))
        Image.fromarray((i_img * 255).astype(np.uint8)).save(path_i_img)
        Image.fromarray((l_img * 255).astype(np.uint8)).save(path_l_img)
