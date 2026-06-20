# def softmax_dice(output, target):
#     '''
#     The dice loss for using softmax activation function
#     :param output: (b, num_class, d, h, w)
#     :param target: (b, d, h, w)
#     :return: softmax dice loss
#     '''
#     loss1 = Dice(output[:, 1, ...], (target == 1).float())
#     loss2 = Dice(output[:, 2, ...], (target == 2).float())
#     loss3 = Dice(output[:, 3, ...], (target == 4).float())

#     return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

# def Dice(output, target, eps=1e-5):
#     target = target.float()
#     num = 2 * (output * target).sum()
#     den = output.sum() + target.sum() + eps
#     return 1.0 - num/den
import torch
import torch.nn as nn
from mmseg.registry import MODELS

@MODELS.register_module()
class SoftmaxDiceLoss(nn.Module):  # 或者 class SoftmaxDiceLoss(nn.Module):
    def __init__(self, eps=1e-5, loss_weight=1.0,loss_name='loss_SD'):
        super().__init__()
        self.eps = eps
        self.loss_weight = loss_weight  # 用于多任务损失加权
        self._loss_name = loss_name

    def forward(self, output, target, **kwargs):
        """
        Args:
            output: (b, num_class, d, h, w)  模型预测的 logits 或概率
            target: (b, d, h, w)  真实标签（类别索引）
        Returns:
            loss: 标量损失值
        """
        # 如果 output 是 logits（未经过 softmax），先计算 softmax
        if output.shape[1] != 1:  # 多分类情况
            output = torch.softmax(output, dim=1)
        
        # 计算各通道的 Dice Loss
        loss1 = self.dice(output[:, 1, ...], (target == 1).float())
        loss2 = self.dice(output[:, 2, ...], (target == 2).float())
        loss3 = self.dice(output[:, 3, ...], (target == 4).float())

        total_loss = loss1 + loss2 + loss3
        return self.loss_weight * total_loss

    def dice(self, output, target):
        """Dice 系数计算"""
        intersection = 2 * (output * target).sum()
        union = output.sum() + target.sum() + self.eps
        return 1 - intersection / union
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