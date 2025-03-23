try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow get_started.md to install MinkowskiEngine.
    ME = None
    pass

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
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

class New3DDetector(BaseModel):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if ME is None:
            raise ImportError(
                'Please follow `get_started.md` to install MinkowskiEngine.`')
        self.voxel_size = bbox_head['voxel_size']
        self.backbone = MODELS.build(backbone)
        self.neck = MODELS.build(neck)
        self.bbox_head=MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feature(self, batch_inputs_dict: dict,
                            batch_data_samples: SampleList):
        # 这里得到的是一个512×1024的depth信息，以及四个尺度的点云，n 256+3 128×256（4w） n 512+3 64*128 8k n 1024+3 32×64 2k n 2048+3 16×32 400的点云
        img = batch_inputs_dict['imgs']
        #img是n 3 512 1024的信息
        points_list,depth = self.backbone(img)
        x=self.neck(points_list)
        x=self.bbox_head(x)
        return x,depth

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
        x,depth = self.extract_feat(batch_inputs)
        losses1 = self.bbox_head.loss(x, batch_data_samples)
        losses2 = self.backbone.loss(depth,batch_data_samples)
        losses=losses1+losses2
        return losses

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        return self.extract_feature(inputs)