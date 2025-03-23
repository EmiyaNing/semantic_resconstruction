"""
## PanoFormer: Panorama Transformer for Indoor 360 Depth Estimation
## Zhijie Shen, Chunyu Lin, Kang Liao, Lang Nie, Zishuo Zheng, Yao Zhao
## https://arxiv.org/abs/2203.09283
## The code is reproducted based on uformer:https://github.com/ZhendongWang6/Uformer
"""
import sys
from scipy.ndimage.filters import maximum_filter
import torch
from shapely.geometry import Polygon
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow get_started.md to install MinkowskiEngine.
    ME = None
    pass
#这一部分是用深度图得到点云然后再转化为minkowski_sparse_tensor
def get_voxels(points,voxel_size):
    """Directly extract features from the backbone+neck.

    Args:
        batch_inputs_dict (dict): The model input dict which includes
            'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.

    Returns:
        tuple[Tensor] | dict:  For outside 3D object detection, we
            typically obtain a tuple of features from the backbone + neck,
            and for inside 3D object detection, usually a dict containing
            features will be obtained.
    """
    coordinates, features = ME.utils.batch_sparse_collate(
        [(p[:, :3] / voxel_size, p[:, 3:]) for p in points],
        device=points[0].device)
    x = ME.SparseTensor(coordinates=coordinates, features=features)
def get_3Dpoints(dep,H,W):
    x,y = np.meshgrid(np.arange(W),np.arange(H))
    theta = ((x + 0.5) / float(W) - 0.5) * 2 * np.pi
    phi = -((y + 0.5) / H - 0.5) * np.pi
    ct,st,cp,sp = np.cos(theta),np.sin(theta),np.cos(phi),np.sin(phi)
    vec = np.array([(cp*ct),(cp*st),(sp)])
    pts = vec * dep
    return pts

    #Output resolution

def get_points_from_depth(depth,img):
    #depth 是n1hw的深度图，里面的值就是正常情况下的米数，然后得到点云
    depth=np.array(depth.cpu())
    img=np.array(img.cpu())
    H,W=depth.shape[2],depth.shape[3]
    pts=get_3Dpoints(depth,H,W)
    pts,img=torch.tensor(pts),torch.tensor(img)
    # pcd.points = op.utility.Vector3dVector(pts.reshape(-1, 3))
    # pcd.colors = op.utility.Vector3dVector(img.reshape(-1, 3) / 255.)
    pts=torch.concat((pts,img),dim=1)
    pts=pts.view(pts.shape[0],pts.shape[1],-1)
    return pts
class StripPooling(nn.Module):
    """
    Reference:
    """

    def __init__(self, in_channels):
        super(StripPooling, self).__init__()
        # self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        # self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool1 = nn.AdaptiveAvgPool2d((1, None))
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))

        self.conv1 = nn.Conv2d(in_channels, in_channels, (1, 3), 1, (0, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, (3, 1), 1, (1, 0), bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.ac = nn.Sigmoid()
        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode="bilinear", align_corners=False)
        x2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode="bilinear", align_corners=False)
        out = self.conv3(x1 + x2)
        out_att = self.ac(out)
        return out_att


#########################################
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x


#########################################
########### feed-forward network #############
class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., flag=0):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=0),
            act_layer())
        # self.hw = StripPooling(hidden_dim)
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))

    def forward(self, x, H, W):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = H

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh * 2)
        # bs,hidden_dim,32x32
        # att = self.hw(x)

        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))

        x = self.dwconv(x)

        # x = x * att

        # x = self.active(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh * 2)

        x = self.linear2(x)

        return x


#########################################
# Downsample Block
#这里是一定要batchnorm的，不然downsample之后得到的方差实在是太大了
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Downsample, self).__init__()
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 1, 1))
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out
#
class DepthDownsample(nn.Module):
    def __init__(self,kernel_size, in_channel=1, out_channel=1):
        super(DepthDownsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)  # B 1 H W()
        return x


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, input_resolution=None):
        super(Upsample, self).__init__()
        self.input_resolution = input_resolution
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.ReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=0),
            act_layer()
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x
#depth和rate的输出是256×512，是要插值一下变成512×1024的
class DepthOutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # for 1024*512
        # x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        # x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        # val=torch.var(x,dim=[2,3])
        return x

class RateOutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
    #这里的插值是可以理解的，因为一个点是不是物体，取决于他旁边的点的信息，如果一个小点的信息是物体，那么旁边的都很有可能是物体
    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        # x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        # x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        # val=torch.var(x,dim=[2,3])
        x = F.interpolate(x, scale_factor=2, mode='nearest')  # for 1024*512
        return x

# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=1, kernel_size=3, stride=1, norm_layer=None, act_layer=None,
                 input_resolution=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, C, H, W)
        # x = F.interpolate(x, scale_factor=2, mode='nearest')  # for 1024*512
        # x = F.pad(x, (3 // 2, 3 // 2, 0, 0), mode='circular')  # width
        # x = F.pad(x, (0, 0, 3 // 2, 3 // 2))
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        # val=torch.var(x,dim=[2,3])
        return x
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pylab import  *
import scipy.misc
from torchvision import models, transforms


def get_k_layer_feature_map(model_layer, k, x):
    with torch.no_grad():
        for index, layer in enumerate(model_layer):  # model的第一个Sequential()是有多层，所以遍�?
            x = layer(x)  # torch.Size([1, 64, 55, 55])生成�?4个通道
            if k == index:
                return x


#  可视化特征图
def show_feature_map(
        feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大�?
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    #upsample = torch.nn.UpsamplingBilinear2d(size=(256, 256))  # 这里进行调整大小
    #feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道�?
    row_num = np.ceil(np.sqrt(feature_map_num))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，�?4个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        #plt.imshow(feature_map[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        #scipy.misc.imsave('feature_map_save//' + str(index) + ".png", feature_map[index - 1])
    plt.show()

def generate_ref_points(width: int,
                        height: int):
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    grid_y = grid_y / (height - 1)
    grid_x = grid_x / (width - 1)

    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    return grid.cuda()


def restore_scale(width: int,
                  height: int,
                  ref_point: torch.Tensor):
    new_point = ref_point.clone().detach()
    new_point[..., 0] = new_point[..., 0] * (width - 1)
    new_point[..., 1] = new_point[..., 1] * (height - 1)

    return new_point
class GridGenerator:
  def __init__(self, height: int, width: int, kernel_size, stride=1):
    self.height = height
    self.width = width
    self.kernel_size = kernel_size  # (Kh, Kw)
    self.stride = stride  # (H, W)

  def createSamplingPattern(self):
    """
    :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
    """
    kerX, kerY = self.createKernel()  # (Kh, Kw)

    # create some values using in generating lat/lon sampling pattern
    rho = np.sqrt(kerX ** 2 + kerY ** 2)
    Kh, Kw = self.kernel_size
    # when the value of rho at center is zero, some lat values explode to `nan`.
    if Kh % 2 and Kw % 2:
      rho[Kh // 2][Kw // 2] = 1e-8

    nu = np.arctan(rho)
    cos_nu = np.cos(nu)
    sin_nu = np.sin(nu)

    stride_h, stride_w = self.stride, self.stride
    h_range = np.arange(0, self.height, stride_h)
    w_range = np.arange(0, self.width, stride_w)

    lat_range = ((h_range / self.height) - 0.5) * np.pi
    lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

    # generate latitude sampling pattern
    lat = np.array([
      np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
    ])  # (H, Kh, Kw)

    lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
    lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # generate longitude sampling pattern
    lon = np.array([
      np.arctan(kerX * sin_nu / (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range
    ])  # (H, Kh, Kw)

    lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
    lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

    # (radian) -> (index of pixel)
    lat = (lat / np.pi + 0.5) * self.height
    lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

    LatLon = np.stack((lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
    LatLon = LatLon.transpose((1, 2, 3, 4, 0))  # (H, Kh, W, Kw, 2) = (H, W,Kh, Kw, (lat, lon))

    H, W, Kh, Kw, d = LatLon.shape
    LatLon = LatLon.reshape((1, H, W, Kh*Kw, d))  # (1, H*Kh, W*Kw, 2)

    return LatLon

  def createKernel(self):
    """
    :return: (Ky, Kx) kernel pattern
    """
    Kh, Kw = self.kernel_size

    delta_lat = np.pi / self.height
    delta_lon = 2 * np.pi / self.width

    range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
    if not Kw % 2:
      range_x = np.delete(range_x, Kw // 2)

    range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
    if not Kh % 2:
      range_y = np.delete(range_y, Kh // 2)

    kerX = np.tan(range_x * delta_lon)
    kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

    return np.meshgrid(kerX, kerY)  # (Kh, Kw)
def genSamplingPattern(h, w, kh, kw, stride=1):
    gridGenerator = GridGenerator(h, w, (kh, kw), stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    # lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
    # lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

    grid = LonLatSamplingPattern#np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      grid = torch.FloatTensor(grid)
      grid.requires_grad = False

    return grid

class PanoSelfAttention(nn.Module):
    def __init__(self, h,
                 d_model,
                 k,
                 last_feat_height,
                 last_feat_width,
                 scales=1,
                 dropout=0.1,
                 need_attn=False):
        """
        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param k: number of keys
        """
        super(PanoSelfAttention, self).__init__()
        #assert h == 8  # currently header is fixed 8 in paper
        assert d_model % h == 0
        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(d_model / h)
        self.h = h

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.scales_hw = []
        for i in range(scales):
            self.scales_hw.append([last_feat_height * 2 ** i,
                                   last_feat_width * 2 ** i])

        self.dropout = None
        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)

        self.k = k
        self.scales = scales
        self.last_feat_height = last_feat_height
        self.last_feat_width = last_feat_width

        self.offset_dims = 2 * self.h * self.k * self.scales
        self.A_dims = self.h * self.k * self.scales

        # 2MLK for offsets MLK for A_mlqk
        self.offset_proj = nn.Linear(d_model, self.offset_dims)
        self.A_proj = nn.Linear(d_model, self.A_dims)

        self.wm_proj = nn.Linear(d_model, d_model)
        self.need_attn = need_attn
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.offset_proj.weight, 0.0)
        torch.nn.init.constant_(self.A_proj.weight, 0.0)

        torch.nn.init.constant_(self.A_proj.bias, 1 / (self.scales * self.k))

        def init_xy(bias, x, y):
            torch.nn.init.constant_(bias[:, 0], float(x))
            torch.nn.init.constant_(bias[:, 1], float(y))

        # caution: offset layout will be  M, L, K, 2
        bias = self.offset_proj.bias.view(self.h, self.scales, self.k, 2)

        # init_xy(bias[0], x=-self.k, y=-self.k)
        # init_xy(bias[1], x=-self.k, y=0)
        # init_xy(bias[2], x=-self.k, y=self.k)
        # init_xy(bias[3], x=0, y=-self.k)
        # init_xy(bias[4], x=0, y=self.k)
        # init_xy(bias[5], x=self.k, y=-self.k)
        # init_xy(bias[6], x=self.k, y=0)
        # init_xy(bias[7], x=self.k, y=self.k)

    def forward(self,
                query: torch.Tensor,
                keys: List[torch.Tensor],
                ref_point: torch.Tensor,
                query_mask: torch.Tensor = None,
                key_masks: Optional[torch.Tensor] = None,
                ):
        """
        :param key_masks:
        :param query_mask:
        :param query: B, H, W, C
        :param keys: List[B, H, W, C]
        :param ref_point: B, H, W, 2
        :return:
        """
        if key_masks is None:
            key_masks = [None] * len(keys)

        assert len(keys) == self.scales

        attns = {'attns': None, 'offsets': None}

        nbatches, query_height, query_width, _ = query.shape

        # B, H, W, C
        query = self.q_proj(query)

        # B, H, W, 2MLK
        offset = self.offset_proj(query)
        # B, H, W, M, 2LK
        offset = offset.view(nbatches, query_height, query_width, self.h, -1)

        # B, H, W, MLK
        A = self.A_proj(query)

        # B, H, W, 1, mask before softmax
        if query_mask is not None:
            query_mask_ = query_mask.unsqueeze(dim=-1)
            _, _, _, mlk = A.shape
            query_mask_ = query_mask_.expand(nbatches, query_height, query_width, mlk)
            A = torch.masked_fill(A, mask=query_mask_, value=float('-inf'))

        # B, H, W, M, LK
        A = A.view(nbatches, query_height, query_width, self.h, -1)
        A = F.softmax(A, dim=-1)

        # mask nan position
        if query_mask is not None:
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            A = torch.masked_fill(A, query_mask_.expand_as(A), 0.0)

        if self.need_attn:
            attns['attns'] = A
            attns['offsets'] = offset

        offset = offset.view(nbatches, query_height, query_width, self.h, self.scales, self.k, 2)
        offset = offset.permute(0, 3, 4, 5, 1, 2, 6).contiguous()
        # B*M, L, K, H, W, 2
        offset = offset.view(nbatches * self.h, self.scales, self.k, query_height, query_width, 2)

        A = A.permute(0, 3, 1, 2, 4).contiguous()
        # B*M, H*W, LK
        A = A.view(nbatches * self.h, query_height * query_width, -1)

        scale_features = []
        for l in range(self.scales):
            feat_map = keys[l]
            _, h, w, _ = feat_map.shape

            key_mask = key_masks[l]

            #ref_point = generate_ref_points(query_width, query_height).repeat(nbatches, 1, 1, 1)

            # B, K, H, W, 2
            reversed_ref_point = ref_point #restore_scale(height=h, width=w, ref_point=ref_point)

            #reversed_ref_point = restore_scale(w, h)

            # B, K, H, W, 2 -> B*M, K, H, W, 2
            reversed_ref_point = reversed_ref_point.repeat(self.h, 1, 1, 1, 1)

            #equi_offset = ref_point_offset.unsqueeze(1)

            # B, h, w, M, C_v
            scale_feature = self.k_proj(feat_map).view(nbatches, h, w, self.h, self.d_k)

            if key_mask is not None:
                # B, h, w, 1, 1
                key_mask = key_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
                key_mask = key_mask.expand(nbatches, h, w, self.h, self.d_k)
                scale_feature = torch.masked_fill(scale_feature, mask=key_mask, value=0)

            # B, M, C_v, h, w
            scale_feature = scale_feature.permute(0, 3, 4, 1, 2).contiguous()
            # B*M, C_v, h, w
            scale_feature = scale_feature.view(-1, self.d_k, h, w)

            k_features = []

            #show_feature_map(scale_feature)



            for k in range(self.k):
                points = reversed_ref_point[:, :, :, k, :] + offset[:, l, k, :, :, :]#+ equi_offset[:, l, :, :, k, :] + offset[:, l, k, :, :, :]
                vgrid_x = 2.0 * points[:, :, :, 1] / max(w - 1, 1) - 1.0
                vgrid_y = 2.0 * points[:, :, :, 0] / max(h - 1, 1) - 1.0
                vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
                #print(points)

                # B*M, C_v, H, W
                feat = F.grid_sample(scale_feature, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)
                #show_feature_map(feat)



                k_features.append(feat)

            # B*M, k, C_v, H, W
            k_features = torch.stack(k_features, dim=1)
            scale_features.append(k_features)

        # B*M, L, K, C_v, H, W
        scale_features = torch.stack(scale_features, dim=1)

        # B*M, H*W, C_v, LK
        scale_features = scale_features.permute(0, 4, 5, 3, 1, 2).contiguous()
        scale_features = scale_features.view(nbatches * self.h, query_height * query_width, self.d_k, -1)

        # B*M, H*W, C_v
        feat = torch.einsum('nlds, nls -> nld', scale_features, A)

        # B*M, H*W, C_v -> B, M, H, W, C_v
        feat = feat.view(nbatches, self.h, query_height, query_width, self.d_k)
        # B, M, H, W, C_v -> B, H, W, M, C_v
        feat = feat.permute(0, 2, 3, 1, 4).contiguous()
        # B, H, W, M, C_v -> B, H, W, M * C_v
        feat = feat.view(nbatches, query_height, query_width, self.d_k * self.h)

        feat = self.wm_proj(feat)
        if self.dropout:
            feat = self.dropout(feat)

        return feat

#########################################
########### LeWinTransformer #############
class PanoformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
                 se_layer=False, ref_point=None, flag=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.ref_point = ref_point  # generate_ref_points(self.input_resolution[1], self.input_resolution[0])
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)

        self.dattn = PanoSelfAttention(num_heads, dim, k=9, last_feat_height=self.input_resolution[0],
                                       last_feat_width=self.input_resolution[1], scales=1, dropout=0, need_attn=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, flag=flag)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x):
        B, L, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # W-MSA/SW-MSA
        x = self.dattn(x, x.unsqueeze(0), self.ref_point.repeat(B, 1, 1, 1, 1))  # nW*B, win_size*win_size, C

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


########### Basic layer of Uformer ################
class BasicPanoformerLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear', token_mlp='leff', se_layer=False, ref_point=None, flag=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            PanoformerBlock(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, win_size=win_size,
                            shift_size=0 if (i % 2 == 0) else win_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            norm_layer=norm_layer, token_projection=token_projection, token_mlp=token_mlp,
                            se_layer=se_layer, ref_point=ref_point)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c//4, out_c),
        )

    def forward(self, x, out_w):
        x = self.layer(x)
        assert out_w % x.shape[3] == 0
        factor = out_w // x.shape[3]
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        x = x[..., factor:-factor]
        return x


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature

def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )

'''
HorizonNet
'''
class HorizonNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self,  use_rnn):
        super(HorizonNet, self).__init__()
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512

        # Encoder
        # Inference channels number from each block of the encoder
        c1, c2, c3, c4 = [256,512,1024,2048]
        c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale
        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)

        # 1D prediction
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=c_last,
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(0.5)
            self.linear = nn.Linear(in_features=2 * self.rnn_hidden_size,
                                    out_features=3 * self.step_cols)
            self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear.bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear.bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        else:
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
            )
            self.linear[-1].bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear[-1].bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear[-1].bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)
    #这里要在panoformer的input的时候用
    def forward(self, x):
        # for i in conv_list:
        #     continue
        feature = self.reduce_height_module(x, 1024//self.step_cols)
        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor
#用来得到inference得到的layout以及通过layout得到depth
PI = float(np.pi)
def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * PI
def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * PI
def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])
def np_x_u_solve_y(x, u, floorW=1024, floorH=512):
    c = (x - floorW / 2 + 0.5) / np.sin(u)
    return -c * np.cos(u) + floorH / 2 - 0.5
def np_y_u_solve_x(y, u, floorW=1024, floorH=512):
    c = -(y - floorH / 2 + 0.5) / np.cos(u)
    return c * np.sin(u) + floorW / 2 - 0.5


def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * PI) + 0.5) * coorW - 0.5
    coory = (-v / PI + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])


def mean_percentile(vec, p1=25, p2=75):
    vmin = np.percentile(vec, p1)
    vmax = np.percentile(vec, p2)
    return vec[(vmin <= vec) & (vec <= vmax)].mean()


def vote(vec, tol):
    vec = np.sort(vec)
    n = np.arange(len(vec))[::-1]
    n = n[:, None] - n[None, :] + 1.0
    l = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

    invalid = (n < len(vec) * 0.4) | (l > tol)
    if (~invalid).sum() == 0 or len(vec) < tol:
        best_fit = np.median(vec)
        p_score = 0
    else:
        l[invalid] = 1e5
        n[invalid] = -1
        score = n
        max_idx = score.argmax()
        max_row = max_idx // len(vec)
        max_col = max_idx % len(vec)
        assert max_col > max_row
        best_fit = vec[max_row:max_col+1].mean()
        p_score = (max_col - max_row + 1) / len(vec)

    l1_score = np.abs(vec - best_fit).mean()

    return best_fit, p_score, l1_score


def get_z1(coory0, coory1, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)
    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    return z1


def np_refine_by_fix_z(coory0, coory1, z0=50, coorH=512):
    '''
    Refine coory1 by coory0
    coory0 are assumed on given plane z
    '''
    v0 = np_coory2v(coory0, coorH)
    v1 = np_coory2v(coory1, coorH)

    c0 = z0 / np.tan(v0)
    z1 = c0 * np.tan(v1)
    z1_mean = mean_percentile(z1)
    v1_refine = np.arctan2(z1_mean, c0)
    coory1_refine = (-v1_refine / PI + 0.5) * coorH - 0.5

    return coory1_refine, z1_mean


def infer_coory(coory0, h, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    c0 = z0 / np.tan(v0)
    z1 = z0 + h
    v1 = np.arctan2(z1, c0)
    return (-v1 / PI + 0.5) * coorH - 0.5


def get_gpid(coorx, coorW):
    gpid = np.zeros(coorW)
    gpid[np.round(coorx).astype(int)] = 1
    gpid = np.cumsum(gpid).astype(int)
    gpid[gpid == gpid[-1]] = 0
    return gpid


def get_gpid_idx(gpid, j):
    idx = np.where(gpid == j)[0]
    if idx[0] == 0 and idx[-1] != len(idx) - 1:
        _shift = -np.where(idx != np.arange(len(idx)))[0][0]
        idx = np.roll(idx, _shift)
    return idx


def gpid_two_split(xy, tpid_a, tpid_b):
    m = np.arange(len(xy)) + 1
    cum_a = np.cumsum(xy[:, tpid_a])
    cum_b = np.cumsum(xy[::-1, tpid_b])
    l1_a = cum_a / m - cum_a / (m * m)
    l1_b = cum_b / m - cum_b / (m * m)
    l1_b = l1_b[::-1]

    score = l1_a[:-1] + l1_b[1:]
    best_split = score.argmax() + 1

    va = xy[:best_split, tpid_a].mean()
    vb = xy[best_split:, tpid_b].mean()

    return va, vb


def _get_rot_rad(px, py):
    if px < 0:
        px, py = -px, -py
    rad = np.arctan2(py, px) * 180 / np.pi
    if rad > 45:
        return 90 - rad
    if rad < -45:
        return -90 - rad
    return -rad


def get_rot_rad(init_coorx, coory, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=5):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)
    xy_cor = []

    rot_rad_suggestions = []
    for j in range(len(init_coorx)):
        pca = PCA(n_components=1)
        pca.fit(xy[gpid == j])
        rot_rad_suggestions.append(_get_rot_rad(*pca.components_[0]))
    rot_rad_suggestions = np.sort(rot_rad_suggestions + [1e9])

    rot_rad = np.mean(rot_rad_suggestions[:-1])
    best_rot_rad_sz = -1
    last_j = 0
    for j in range(1, len(rot_rad_suggestions)):
        if rot_rad_suggestions[j] - rot_rad_suggestions[j-1] > tol:
            last_j = j
        elif j - last_j > best_rot_rad_sz:
            rot_rad = rot_rad_suggestions[last_j:j+1].mean()
            best_rot_rad_sz = j - last_j

    dx = int(round(rot_rad * 1024 / 360))
    return dx, rot_rad


def gen_ww_cuboid(xy, gpid, tol):
    xy_cor = []
    assert len(np.unique(gpid)) == 4

    # For each part seperated by wall-wall peak, voting for a wall
    for j in range(4):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score})

    # Sanity fallback
    scores = [0, 0]
    for j in range(4):
        if xy_cor[j]['type'] == 0:
            scores[j % 2] += xy_cor[j]['score']
        else:
            scores[j % 2] -= xy_cor[j]['score']
    if scores[0] > scores[1]:
        xy_cor[0]['type'] = 0
        xy_cor[1]['type'] = 1
        xy_cor[2]['type'] = 0
        xy_cor[3]['type'] = 1
    else:
        xy_cor[0]['type'] = 1
        xy_cor[1]['type'] = 0
        xy_cor[2]['type'] = 1
        xy_cor[3]['type'] = 0

    return xy_cor


def gen_ww_general(init_coorx, xy, gpid, tol):
    xy_cor = []
    assert len(init_coorx) == len(np.unique(gpid))

    # Candidate for each part seperated by wall-wall boundary
    for j in range(len(init_coorx)):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        u0 = np_coorx2u(init_coorx[(j - 1 + len(init_coorx)) % len(init_coorx)])
        u1 = np_coorx2u(init_coorx[j])
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score, 'action': 'ori', 'gpid': j, 'u0': u0, 'u1': u1, 'tbd': True})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score, 'action': 'ori', 'gpid': j, 'u0': u0, 'u1': u1, 'tbd': True})

    # Construct wall from highest score to lowest
    while True:
        # Finding undetermined wall with highest score
        tbd = -1
        for i in range(len(xy_cor)):
            if xy_cor[i]['tbd'] and (tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                tbd = i
        if tbd == -1:
            break

        # This wall is determined
        xy_cor[tbd]['tbd'] = False
        p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
        n_idx = (tbd + 1) % len(xy_cor)

        num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']

        # Two adjacency walls are not determined yet => not special case
        if num_tbd_neighbor == 2:
            continue

        # Only one of adjacency two walls is determine => add now or later special case
        if num_tbd_neighbor == 1:
            if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                    (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                # Current wall is different from one determined adjacency wall
                if xy_cor[tbd]['score'] >= -1:
                    # Later special case, add current to tbd
                    xy_cor[tbd]['tbd'] = True
                    xy_cor[tbd]['score'] -= 100
                else:
                    # Fallback: forced change the current wall or infinite loop
                    if not xy_cor[p_idx]['tbd']:
                        insert_at = tbd
                        if xy_cor[p_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 0
                    else:
                        insert_at = n_idx
                        if xy_cor[n_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 0
                    new_add = {'type': new_type, 'val': new_val, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False}
                    xy_cor.insert(insert_at, new_add)
            continue

        # Below checking special case
        if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
            # Two adjacency walls are same type, current wall should be differen type
            if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                # Fallback: three walls with same type => forced change the middle wall
                xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                xy_cor[tbd]['action'] = 'forced change'
                xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'], xy_cor[tbd]['type']].mean()
        else:
            # Two adjacency walls are different type => add one
            tp0 = xy_cor[n_idx]['type']
            tp1 = xy_cor[p_idx]['type']
            if xy_cor[p_idx]['type'] == 0:
                val0 = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            else:
                val0 = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            new_add = [
                {'type': tp0, 'val': val0, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False},
                {'type': tp1, 'val': val1, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False},
            ]
            xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd+1:]

    return xy_cor


def gen_ww(init_coorx, coory, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=3, force_cuboid=True):
    gpid = get_gpid(init_coorx, coorW)
    coor = np.hstack([np.arange(coorW)[:, None], coory[:, None]])
    xy = np_coor2xy(coor, z, coorW, coorH, floorW, floorH)

    # Generate wall-wall
    if force_cuboid:
        xy_cor = gen_ww_cuboid(xy, gpid, tol)
    else:
        xy_cor = gen_ww_general(init_coorx, xy, gpid, tol)

    # Ceiling view to normal view
    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        if xy_cor[j]['type'] == 1:
            cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
        else:
            cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
    cor = np_xy2coor(np.array(cor), z, coorW, coorH, floorW, floorH)
    cor = np.roll(cor, -2 * cor[::2, 0].argmin(), axis=0)

    return cor, xy_cor
def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv


@functools.lru_cache()
def _uv_tri(w, h):
    uv = uv_meshgrid(w, h)
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    tan_v = np.tan(uv[..., 1])
    return sin_u, cos_u, tan_v


def uv_tri(w, h):
    sin_u, cos_u, tan_v = _uv_tri(w, h)
    return sin_u.copy(), cos_u.copy(), tan_v.copy()


def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi


def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi


def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5


def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5


def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

#现在理解了
def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    if p1[0] == p2[0]:
        return np.array([p1, p2], np.float32)

    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs, h)

    return np.stack([coorxs, coorys], axis=-1)


def pano_stretch(img, corners, kx, ky, order=1):
    '''
    img:     [H, W, C]
    corners: [N, 2] in image coordinate (x, y) format
    kx:      Stretching along front-back direction
    ky:      Stretching along left-right direction
    order:   Interpolation order. 0 for nearest-neighbor. 1 for bilinear.
    '''

    # Process image
    sin_u, cos_u, tan_v = uv_tri(img.shape[1], img.shape[0])
    u0 = np.arctan2(sin_u * kx / ky, cos_u)
    v0 = np.arctan(tan_v * np.sin(u0) / sin_u * ky)

    refx = (u0 / (2 * np.pi) + 0.5) * img.shape[1] - 0.5
    refy = (v0 / np.pi + 0.5) * img.shape[0] - 0.5

    # [TODO]: using opencv remap could probably speedup the process a little
    stretched_img = np.stack([
        map_coordinates(img[..., i], [refy, refx], order=order, mode='wrap')
        for i in range(img.shape[-1])
    ], axis=-1)

    # Process corners
    corners_u0 = coorx2u(corners[:, 0], img.shape[1])
    corners_v0 = coory2v(corners[:, 1], img.shape[0])
    corners_u = np.arctan2(np.sin(corners_u0) * ky / kx, np.cos(corners_u0))
    C2 = (np.sin(corners_u0) * ky)**2 + (np.cos(corners_u0) * kx)**2
    corners_v = np.arctan2(
            np.sin(corners_v0),
            np.cos(corners_v0) * np.sqrt(C2))

    cornersX = u2coorx(corners_u, img.shape[1])
    cornersY = v2coory(corners_v, img.shape[0])
    stretched_corners = np.stack([cornersX, cornersY], axis=-1)

    return stretched_img, stretched_corners

def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first)*2-1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys
def cor_2_1d(cor, H, W):
    bon_ceil_x, bon_ceil_y = [], []
    bon_floor_x, bon_floor_y = [], []
    n_cor = len(cor)
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2],
                                              cor[(i*2+2) % n_cor],
                                              z=-50, w=W, h=H)
        bon_ceil_x.extend(xys[:, 0])
        bon_ceil_y.extend(xys[:, 1])
    for i in range(n_cor // 2):
        xys = pano_connect_points(cor[i*2+1],
                                              cor[(i*2+3) % n_cor],
                                              z=50, w=W, h=H)
        bon_floor_x.extend(xys[:, 0])
        bon_floor_y.extend(xys[:, 1])
    bon_ceil_x, bon_ceil_y = sort_xy_filter_unique(bon_ceil_x, bon_ceil_y, y_small_first=True)
    bon_floor_x, bon_floor_y = sort_xy_filter_unique(bon_floor_x, bon_floor_y, y_small_first=False)
    bon = np.zeros((2, W))
    bon[0] = np.interp(np.arange(W), bon_ceil_x, bon_ceil_y, period=W)
    bon[1] = np.interp(np.arange(W), bon_floor_x, bon_floor_y, period=W)
    bon = ((bon + 0.5) / H - 0.5) * np.pi
    return bon


def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth这里可以改成1.5，理论上会让指标更好，现在尝试1.43,1.56(所有深度图的均值)
    floor_h = 1.56
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth
def inference(y_bon_,y_cor_,force_cuboid=False, force_raw=False, min_v=None, r=0.05):
    '''
    net   : the trained HorizonNet
    x     : tensor in shape [1, 3, 512, 1024]
    flip  : fliping testing augmentation
    rotate: horizontal rotation testing augmentation
    '''
    H, W = tuple((512,1024))
    y_bon_=np.array(y_bon_.cpu())
    y_cor_=torch.sigmoid(y_cor_.cpu())
    y_cor_ = np.array(y_cor_)
    # Network feedforward (with testing augmentation)
    y_bon_ = (y_bon_[0] / np.pi + 0.5) * H - 0.5
    y_bon_[0] = np.clip(y_bon_[0], 1, H/2-1)
    y_bon_[1] = np.clip(y_bon_[1], H/2+1, H-2)
    y_cor_ = y_cor_[0, 0]
    # Init floor/ceil plane 这个是将底部的高度统一化为z_mean
    z0 = 50
    _, z1 = np_refine_by_fix_z(*y_bon_, z0)
    cor = np.stack([np.arange(1024), y_bon_[0]], 1)
    cor = np.hstack([cor, infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor)*2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j*2] = cor[j, 0], cor[j, 1]
        cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]
    # 不能Normalized to [0, 1]要保留到512×1024
    depth, floor_mask, ceil_mask, wall_mask = layout_2_depth(cor_id, H, W, return_mask=True)
    return torch.from_numpy(depth)
########### Uformer ################这里是整体网络的backbone，通过输入一个全景图，得到一个depth和四个stage的voxels
@MODELS.register_module()
class Panoformer(nn.Module):
    def __init__(self, img_size=256, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='leff', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        # self.ref_point256x512 = genSamplingPattern(256, 512, 3, 3).cuda()#torch.load("network6/Equioffset256x512.pth")
        self.ref_point256x512 = genSamplingPattern(256, 512, 3,
                                                   3).cuda()  # torch.load("network6/Equioffset256x512.pth")
        self.ref_point128x256 = genSamplingPattern(128, 256, 3,
                                                   3).cuda()  # torch.load("network6/Equioffset128x256.pth")
        self.ref_point64x128 = genSamplingPattern(64, 128, 3, 3).cuda()  # torch.load("network6/Equioffset64x128.pth")
        self.ref_point32x64 = genSamplingPattern(32, 64, 3, 3).cuda()  ##torch.load("network6/Equioffset32x64.pth")
        self.ref_point16x32 = genSamplingPattern(16, 32, 3, 3).cuda()  # torch.load("network6/Equioffset16x32.pth")
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        # self.pre_block = PreprocBlock(in_channels=3, out_channels=64, kernel_size_lst=[[3, 9], [5, 11], [5, 7], [7, 7]])

        self.input_proj = InputProj(in_channel=in_chans, out_channel=embed_dim, kernel_size=3, stride=2,
                                    act_layer=nn.GELU)  # stride = 2 for 1024*512
        # self.conv_in=nn.Sequential(
        #     nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, stride=2, padding=1),
        #     nn.GELU()
        # )
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=1, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))
        self.output_proj_0 = OutputProj(in_channel=2 * embed_dim, out_channel=8 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // 2, img_size // 2 * 2))
        self.output_proj_1 = OutputProj(in_channel=4 * embed_dim, out_channel=16 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // (2 * 2), img_size // (2 ** 2) * 2))
        self.output_proj_2 = OutputProj(in_channel=8 * embed_dim, out_channel=32 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // (2 ** 3), img_size // (2 ** 3) * 2))
        self.output_proj_3 = OutputProj(in_channel=16 * embed_dim, out_channel=64 * embed_dim, kernel_size=3, stride=1,
                                        input_resolution=(img_size // (2 ** 4), img_size // (2 ** 4) * 2))
        self.item_rate = RateOutputProj(in_channel=2 * embed_dim, out_channel=1, kernel_size=3, stride=1,
                                         input_resolution=(img_size, img_size * 2))
        self.output_proj = DepthOutputProj(in_channel=2 * embed_dim, out_channel=1, kernel_size=3, stride=1,
                                      input_resolution=(img_size, img_size * 2))
        # Encoder
        self.encoderlayer_0 = BasicPanoformerLayer(dim=embed_dim,
                                                   output_dim=embed_dim,
                                                   input_resolution=(img_size, img_size * 2),
                                                   depth=depths[0],
                                                   num_heads=num_heads[0],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[int(sum(depths[:0])):int(sum(depths[:1]))],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point256x512, flag=0)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2, input_resolution=(img_size, img_size * 2))
        self.encoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 2,
                                                   output_dim=embed_dim * 2,
                                                   input_resolution=(img_size // 2, img_size * 2 // 2),
                                                   depth=depths[1],
                                                   num_heads=num_heads[1],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point128x256, flag=0)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.encoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4,
                                                   output_dim=embed_dim * 4,
                                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                   depth=depths[2],
                                                   num_heads=num_heads[2],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point64x128, flag=0)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8,
                                     input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.encoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 8,
                                                   output_dim=embed_dim * 8,
                                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                   depth=depths[3],
                                                   num_heads=num_heads[3],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point32x64, flag=0)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16,
                                     input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))

        # Bottleneck
        self.conv = BasicPanoformerLayer(dim=embed_dim * 16,
                                         output_dim=embed_dim * 16,
                                         input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)),
                                         depth=depths[4],
                                         num_heads=num_heads[4],
                                         win_size=win_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=conv_dpr,
                                         norm_layer=norm_layer,
                                         use_checkpoint=use_checkpoint,
                                         token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer,
                                         ref_point=self.ref_point16x32, flag=0)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8,
                                   input_resolution=(img_size // (2 ** 4), img_size * 2 // (2 ** 4)))
        self.decoderlayer_0 = BasicPanoformerLayer(dim=embed_dim * 16,
                                                   output_dim=embed_dim * 16,
                                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)),
                                                   depth=depths[5],
                                                   num_heads=num_heads[5],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[:depths[5]],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point32x64, flag=1)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4,
                                   input_resolution=(img_size // (2 ** 3), img_size * 2 // (2 ** 3)))
        self.decoderlayer_1 = BasicPanoformerLayer(dim=embed_dim * 8,
                                                   output_dim=embed_dim * 8,
                                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)),
                                                   depth=depths[6],
                                                   num_heads=num_heads[6],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point64x128, flag=1)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2,
                                   input_resolution=(img_size // (2 ** 2), img_size * 2 // (2 ** 2)))
        self.decoderlayer_2 = BasicPanoformerLayer(dim=embed_dim * 4,
                                                   output_dim=embed_dim * 4,
                                                   input_resolution=(img_size // 2, img_size * 2 // 2),
                                                   depth=depths[7],
                                                   num_heads=num_heads[7],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[5:7]):sum(depths[5:8])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point128x256, flag=1)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim, input_resolution=(img_size // 2, img_size * 2 // 2))
        self.decoderlayer_3 = BasicPanoformerLayer(dim=embed_dim * 2,
                                                   output_dim=embed_dim * 2,
                                                   input_resolution=(img_size, img_size * 2),
                                                   depth=depths[8],
                                                   num_heads=num_heads[8],
                                                   win_size=win_size,
                                                   mlp_ratio=self.mlp_ratio,
                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                   drop=drop_rate, attn_drop=attn_drop_rate,
                                                   drop_path=dec_dpr[sum(depths[5:8]):sum(depths[5:9])],
                                                   norm_layer=norm_layer,
                                                   use_checkpoint=use_checkpoint,
                                                   token_projection=token_projection, token_mlp=token_mlp,
                                                   se_layer=se_layer, ref_point=self.ref_point256x512, flag=1)
        self.lstmdecoder=HorizonNet(use_rnn=True)
        self.depth_downsample0 = DepthDownsample(kernel_size=4)
        self.depth_downsample1 = DepthDownsample(kernel_size=8)
        self.depth_downsample2 = DepthDownsample(kernel_size=16)
        self.depth_downsample3 = DepthDownsample(kernel_size=32)
        self.apply(self._init_weights)
        self.freezen_list=[self.lstmdecoder,self.input_proj,self.output_proj_0,self.output_proj_1,self.output_proj_2,
                           self.output_proj_3,self.encoderlayer_0,self.encoderlayer_1,self.encoderlayer_2,self.encoderlayer_3,
                           self.dowsample_0,self.dowsample_1,self.dowsample_2,self.dowsample_3]
        for submodel in self.freezen_list:
            for name, param in submodel.named_parameters():
                param.requires_grad = False
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x):
        # Input Projection
        # y = self.pre_block(x)
        y = self.input_proj(x)
        y = self.pos_drop(y)

        # Encoder
        conv0 = self.encoderlayer_0(y)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2)
        pool3 = self.dowsample_3(conv3)
        #使用encoder的输出得到layout并且得到对应的depth
        layout_feature = []
        layout_feature.append(self.output_proj_0(pool0))
        layout_feature.append(self.output_proj_1(pool1))
        layout_feature.append(self.output_proj_2(pool2))
        layout_feature.append(self.output_proj_3(pool3))
        bon,cor=self.lstmdecoder(layout_feature)
        layout_depth=inference(bon,cor,force_raw=True).cuda()
        # Bottleneck
        conv4 = self.conv(pool3)
        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3)

        # Output Projection 获得了depth
        item_rate = self.item_rate(deconv3)
        depth1 = self.output_proj(deconv3)
        depth2 = layout_depth
        y=depth1*item_rate+depth2*(1-item_rate)
        outputs = {}
        outputs["pred_depth"] = y
        #从深度图得到四个尺度的voxels,先concat每个角度的深度图和feature，然后得到每个深度图的xyz信息，然后再重新concat回去，变成点云
        depth_pool0 = self.depth_downsample0(y)
        depth_pool1 = self.depth_downsample1(y)
        depth_pool2 = self.depth_downsample2(y)
        depth_pool3 = self.depth_downsample3(y)
        voxels_list=[]
        points0 = (get_points_from_depth(depth_pool0, pool0)).cuda()
        points1 = (get_points_from_depth(depth_pool1, pool1)).cuda()
        points2 = (get_points_from_depth(depth_pool2, pool2)).cuda()
        points3 = (get_points_from_depth(depth_pool3, pool3)).cuda()
        voxels_list.append(get_voxels(points0, voxel_size=0.01))
        voxels_list.append(get_voxels(points1, voxel_size=0.01))
        voxels_list.append(get_voxels(points2, voxel_size=0.01))
        voxels_list.append(get_voxels(points3, voxel_size=0.01))
        outputs["voxels"]=voxels_list
        return outputs



