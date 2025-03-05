# import torchvision
# from distutils.version import LooseVersion
# is_old_version = LooseVersion(torchvision.__version__) < LooseVersion("0.13.0")
# is_new_version = LooseVersion(torchvision.__version__) >= LooseVersion("0.13.0")

# if is_new_version:
#     from torchvision.models import ResNet50_Weights, DenseNet121_Weights
# else:
#     pass
# ENCODER_RESNET = [
#     'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
#     'resnext50_32x4d', 'resnext101_32x8d'
# ]
# ENCODER_DENSENET = [
#     'densenet121', 'densenet169', 'densenet161', 'densenet201'
# ]
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule, Sequential
from torch import nn as nn
import torch.nn.functional as F
import functools
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModel
from typing import Tuple

from mmdet.models.detectors.single_stage import SingleStageDetector
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmengine.utils import is_list_of

from mmdet3d.utils import OptInstanceList
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
class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Defaults to 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Defaults to False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Defaults to None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Defaults to dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Defaults to True
    """

    def __init__(self,
                 block: BaseModule,
                 inplanes: int,
                 planes: int,
                 num_blocks: int,
                 stride: int = 1,
                 avg_down: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 downsample_first: bool = True,
                 **kwargs) -> None:
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg ,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super().__init__(*layers)
class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class Resnet(BaseModule):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 # dcn=None,
                 # stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=False,
                 pretrained=None,
                 init_cfg=None):
        super(Resnet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        # if depth not in self.arch_settings:
        #     raise KeyError(f'invalid depth {depth} for resnet')
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        #可变性卷积如何实现之后再说
        # self.dcn = dcn
        # self.stage_with_dcn = stage_with_dcn
        # if dcn is not None:
        #     assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            # dcn = self.dcn if self.stage_with_dcn[i] else None
            # if plugins is not None:
            #     stage_plugins = self.make_stage_plugins(plugins, i)
            # else:
            #     stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                # dcn=dcn,
                # plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        #out的输出是256*128*256,512*64*128,1024*32*64,2048*16*32,然后经过h/16,w变为w_out，即为256，我们想要的东西channel变成channel/8


        return tuple(outs)

    # def train(self, mode=True):
    #     """Convert the model into training mode while keep normalization layer
    #     freezed."""
    #     super(Resnet, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''

    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 2),
            ConvCompressH(in_c // 2, in_c // 4),
            ConvCompressH(in_c // 4, out_c),
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
            GlobalHeightConv(c1, c1 // out_scale),
            GlobalHeightConv(c2, c2 // out_scale),
            GlobalHeightConv(c3, c3 // out_scale),
            GlobalHeightConv(c4, c4 // out_scale),
        ])

    def forward(self, conv_list, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        feature = torch.cat([
            f(x, out_w).reshape(bs, -1, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature

@MODELS.register_module()
class HorizonNet(BaseModel):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType=None,
                 bbox_head: ConfigType=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor,init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        # self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs_dict (dict): Contains 'img' key
                with image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        batch_imgs = batch_inputs_dict['imgs']
        x = self.backbone(batch_imgs)
        if self.with_neck:
            x = self.neck(x)
        return x
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses
    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        return self.extract_feat(inputs)

@MODELS.register_module()
class HorizonNetwithResnet(BaseModule):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])
    #这里的init是horizonnet的init，是在backbone里面实现的
    def __init__(self,use_rnn,depth,num_stages,out_indices,frozen_stages,norm_cfg,norm_eval,style):
        super(HorizonNetwithResnet,self).__init__()
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 4
        self.rnn_hidden_size = 512
        self.style=style
        self.feature_extractor=Resnet(depth=50,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style=self.style,
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 # dcn=None,
                 # stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=False,
                 pretrained=None,
                 init_cfg=None)

        # Encoder
        # if is_old_version:
        #     if backbone.startswith('res'):
        #         self.feature_extractor = Resnet(backbone, pretrained=True)
        #     elif backbone.startswith('dense'):
        #         self.feature_extractor = Densenet(backbone, pretrained=True)
        #     else:
        #         raise NotImplementedError()
        # elif is_new_version:
        #     if backbone.startswith('res'):
        #         self.feature_extractor = Resnet(backbone, weights=ResNet50_Weights.IMAGENET1K_V1)
        #     elif backbone.startswith('dense'):
        #         self.feature_extractor = Densenet(backbone, weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1 * 8 + c2 * 4 + c3 * 2 + c4 * 1) // self.out_scale

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
            self.linear.bias.data[0 * self.step_cols:1 * self.step_cols].fill_(-1)
            self.linear.bias.data[1 * self.step_cols:2 * self.step_cols].fill_(-0.478)
            self.linear.bias.data[2 * self.step_cols:3 * self.step_cols].fill_(0.425)
        else:
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
            )
            self.linear[-1].bias.data[0 * self.step_cols:1 * self.step_cols].fill_(-1)
            self.linear[-1].bias.data[1 * self.step_cols:2 * self.step_cols].fill_(-0.478)
            self.linear[-1].bias.data[2 * self.step_cols:3 * self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()
        x = self._prepare_x(x)
        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)

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

