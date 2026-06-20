#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from torch import nn
from mmseg.registry import MODELS
from monai.losses import DiceCELoss
import numpy as np
from monai.networks import one_hot
from typing import Callable, List, Optional, Sequence, Union

num_classes = 9
class_weights = torch.ones(num_classes)
class_weights[6] = 5.0  # 第七类权重提升

feat_size = [48,96,192,384]
# 修改后的DiceCELoss
dice_loss = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    squared_pred=True,
    smooth_nr=0.0,
    smooth_dr=1e-6,
    ce_weight=class_weights,  # 交叉熵类别权重
)
loss_func = dice_loss

net_numpool = len(feat_size[:3])  # Except for hidden feature. Todo: defined by args
weights = np.array([1 / (2 ** (i+1)) if i != 0 else 1 / (2 ** i) for i in range(net_numpool)])  # 4 stages                             
weights = weights / weights.sum()
ds_loss_weights = weights

@MODELS.register_module()
class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss=loss_func, weight_factors=ds_loss_weights,loss_name='loss_MOL'):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self._loss_name = loss_name

    def forward(self, x, y,weight=None,ignore_index=0):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        # l = weights[0] * self.loss(x[0], y[0])
        mid_loss = self.loss(x[0], y[0])
        l = torch.mul(weights[0], mid_loss)
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

