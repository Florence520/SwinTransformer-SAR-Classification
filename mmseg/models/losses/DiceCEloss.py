
import torch
from torch import nn
from mmseg.registry import MODELS
from monai.losses import DiceCELoss
import numpy as np
dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6,
    )
loss_func = dice_loss

@MODELS.register_module()
class DiceCEloss(nn.Module):
    def __init__(self, loss=loss_func, loss_name='loss_DCE'):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(DiceCEloss, self).__init__()

        self.loss = loss
        self._loss_name = loss_name

    def forward(self, x, y,weight=None,ignore_index=0):
        y = y.unsqueeze(1)
        mid_loss = self.loss(x, y)

        return mid_loss
    
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
