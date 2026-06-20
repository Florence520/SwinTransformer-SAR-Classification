from .cross_entropy_loss import CrossEntropyLoss
from .accuracy import accuracy
from .deep_supervision import MultipleOutputLoss2
from .softmax_dice import SoftmaxDiceLoss
from .DiceCEloss import DiceCEloss
__all = ['CrossEntropyLoss','accuracy','MultipleOutputLoss2','SoftmaxDiceLoss','DiceCEloss']