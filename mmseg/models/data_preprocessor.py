# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import Mul_stack_batch,stack_batch


@MODELS.register_module()
class SegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class MulSegDataPreProcessor(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
            self,
            mean1: Sequence[Number] = None,
            mean2: Sequence[Number] = None,
            std1: Sequence[Number] = None,
            std2: Sequence[Number] = None,
            size: Optional[tuple] = None,
            size_divisor: Optional[int] = None,
            pad_val: Number = 0,
            numtimes=0,
            seg_pad_val: Number = 255,
            bgr_to_rgb: bool = False,
            rgb_to_bgr: bool = False,
            batch_augments: Optional[List[dict]] = None,
            test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.numtimes = numtimes

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean1 is not None and std1 is not None and mean2 is not None and std2 is not None:
            # 确保所有必要的参数都已指定
            assert len(mean1) == 2, "mean1 should have length 2"
            assert len(std1) == 2, "std1 should have length 2"
            assert len(mean2) == 6, "mean2 should have length 6"
            assert len(std2) == 6, "std2 should have length 6"

            # 启用归一化
            self._enable_normalize = True

            # 注册均值和标准差作为缓冲区
            # 对于多通道数据，我们不需要额外的空间维度（即不需要view(-1, 1, 1)），除非后续操作需要
            self.register_buffer('mean1', torch.tensor(mean1), False)  # 直接注册为长度为2的向量
            self.register_buffer('std1', torch.tensor(std1), False)  # 直接注册为长度为2的向量
            self.register_buffer('mean2', torch.tensor(mean2), False)  # 直接注册为长度为6的向量
            self.register_buffer('std2', torch.tensor(std2), False)  # 直接注册为长度为6的向量
        else:
            # 如果任何必需的参数未指定，则不启用归一化
            self._enable_normalize = False

            # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        inputs1 = data['inputs1']
        inputs2 = data['inputs2']
        data_samples = data.get('data_samples', None)
        # if len(data_samples) == 2:
        #     s2_cloud_path1 = data_samples[0].img_p2_path
        #     s2_cloud_path2 = data_samples[1].img_p2_path
        #     s2_cloud = [s2_cloud_path1,s2_cloud_path2]
        # else:
        #     s2_cloud_path1 = data_samples[0].img_p2_path
        #     s2_cloud = [s2_cloud_path1]
        # TODO: whether normalize should be after stack_batch

        inputs1 = [_input.float() for _input in inputs1]
        inputs2 = [_input.float() for _input in inputs2]
        if self._enable_normalize:
            inputs1 = [(_input - self.mean1) / self.std1 for _input in inputs1]
            inputs2 = [(_input - self.mean2) / self.std2 for _input in inputs2]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs1, inputs2, data_samples = Mul_stack_batch(
                inputs1=inputs1,
                inputs2=inputs2,
                # s2_cloud=s2_cloud,
                data_samples=data_samples,
                size=self.size,
                numtimes=self.numtimes,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)


        else:
            img1_size = inputs1[0].shape[2:]
            assert all(input_.shape[2:] == img1_size for input_ in inputs1), \
                'The image size in a batch should be the same.'

            img2_size = inputs2[0].shape[2:]
            assert all(input_.shape[2:] == img2_size for input_ in inputs2), \
                'The image size in a batch should be the same.'

            # pad images when testing
            if self.test_cfg:
                inputs1, inputs2, padded_samples = Mul_stack_batch(
                    inputs1=inputs1,
                    inputs2=inputs2,
                    # s2_cloud=s2_cloud,
                    size=self.test_cfg.get('size', None),
                    numtimes=self.numtimes,
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs1 = torch.stack(inputs1, dim=0)
                inputs2 = torch.stack(inputs2, dim=0)

        return dict(inputs1=inputs1, inputs2=inputs2, data_samples=data_samples)
    


# @MODELS.register_module()
# class MulSegDataPreProcessor(BaseDataPreprocessor):
#     """Image pre-processor for segmentation tasks.

#     Comparing with the :class:`mmengine.ImgDataPreprocessor`,

#     1. It won't do normalization if ``mean`` is not specified.
#     2. It does normalization and color space conversion after stacking batch.
#     3. It supports batch augmentations like mixup and cutmix.


#     It provides the data pre-processing as follows

#     - Collate and move data to the target device.
#     - Pad inputs to the input size with defined ``pad_val``, and pad seg map
#         with defined ``seg_pad_val``.
#     - Stack inputs to batch_inputs.
#     - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
#     - Normalize image with defined std and mean.
#     - Do batch augmentations like Mixup and Cutmix during training.

#     Args:
#         mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
#             Defaults to None.
#         std (Sequence[Number], optional): The pixel standard deviation of
#             R, G, B channels. Defaults to None.
#         size (tuple, optional): Fixed padding size.
#         size_divisor (int, optional): The divisor of padded size.
#         pad_val (float, optional): Padding value. Default: 0.
#         seg_pad_val (float, optional): Padding value of segmentation map.
#             Default: 255.
#         padding_mode (str): Type of padding. Default: constant.
#             - constant: pads with a constant value, this value is specified
#               with pad_val.
#         bgr_to_rgb (bool): whether to convert image from BGR to RGB.
#             Defaults to False.
#         rgb_to_bgr (bool): whether to convert image from RGB to RGB.
#             Defaults to False.
#         batch_augments (list[dict], optional): Batch-level augmentations
#         test_cfg (dict, optional): The padding size config in testing, if not
#             specify, will use `size` and `size_divisor` params as default.
#             Defaults to None, only supports keys `size` or `size_divisor`.
#     """

#     def __init__(
#             self,
#             mean1: Sequence[Number] = None,
#             mean2: Sequence[Number] = None,
#             std1: Sequence[Number] = None,
#             std2: Sequence[Number] = None,
#             size: Optional[tuple] = None,
#             size_divisor: Optional[int] = None,
#             pad_val: Number = 0,
#             numtimes=0,
#             seg_pad_val: Number = 255,
#             bgr_to_rgb: bool = False,
#             rgb_to_bgr: bool = False,
#             batch_augments: Optional[List[dict]] = None,
#             test_cfg: dict = None,
#     ):
#         super().__init__()
#         self.size = size
#         self.size_divisor = size_divisor
#         self.pad_val = pad_val
#         self.seg_pad_val = seg_pad_val
#         self.numtimes = numtimes

#         assert not (bgr_to_rgb and rgb_to_bgr), (
#             '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
#         self.channel_conversion = rgb_to_bgr or bgr_to_rgb

#         if mean1 is not None and std1 is not None and mean2 is not None and std2 is not None:
#             # 确保所有必要的参数都已指定
#             assert len(mean1) == 2, "mean1 should have length 2"
#             assert len(std1) == 2, "std1 should have length 2"
#             assert len(mean2) == 6, "mean2 should have length 6"
#             assert len(std2) == 6, "std2 should have length 6"

#             # 启用归一化
#             self._enable_normalize = True

#             # 注册均值和标准差作为缓冲区
#             # 对于多通道数据，我们不需要额外的空间维度（即不需要view(-1, 1, 1)），除非后续操作需要
#             self.register_buffer('mean1', torch.tensor(mean1), False)  # 直接注册为长度为2的向量
#             self.register_buffer('std1', torch.tensor(std1), False)  # 直接注册为长度为2的向量
#             self.register_buffer('mean2', torch.tensor(mean2), False)  # 直接注册为长度为6的向量
#             self.register_buffer('std2', torch.tensor(std2), False)  # 直接注册为长度为6的向量
#         else:
#             # 如果任何必需的参数未指定，则不启用归一化
#             self._enable_normalize = False

#             # TODO: support batch augmentations.
#         self.batch_augments = batch_augments

#         # Support different padding methods in testing
#         self.test_cfg = test_cfg

#     def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
#         """Perform normalization、padding and bgr2rgb conversion based on
#         ``BaseDataPreprocessor``.

#         Args:
#             data (dict): data sampled from dataloader.
#             training (bool): Whether to enable training time augmentation.

#         Returns:
#             Dict: Data in the same format as the model input.
#         """
#         data = self.cast_data(data)  # type: ignore
#         inputs1 = data['inputs1']
#         inputs2 = data['inputs2']
#         data_samples = data.get('data_samples', None)
#         # TODO: whether normalize should be after stack_batch

#         inputs1 = [_input.float() for _input in inputs1]
#         inputs2 = [_input.float() for _input in inputs2]
#         if self._enable_normalize:
#             inputs1 = [(_input - self.mean1) / self.std1 for _input in inputs1]
#             inputs2 = [(_input - self.mean2) / self.std2 for _input in inputs2]

#         if training:
#             assert data_samples is not None, ('During training, ',
#                                               '`data_samples` must be define.')
#             inputs1, inputs2, data_samples = Mul_stack_batch(
#                 inputs1=inputs1,
#                 inputs2=inputs2,
#                 data_samples=data_samples,
#                 size=self.size,
#                 numtimes=self.numtimes,
#                 size_divisor=self.size_divisor,
#                 pad_val=self.pad_val,
#                 seg_pad_val=self.seg_pad_val)


#         else:
#             img1_size = inputs1[0].shape[2:]
#             assert all(input_.shape[2:] == img1_size for input_ in inputs1), \
#                 'The image size in a batch should be the same.'

#             img2_size = inputs2[0].shape[2:]
#             assert all(input_.shape[2:] == img2_size for input_ in inputs2), \
#                 'The image size in a batch should be the same.'

#             # pad images when testing
#             if self.test_cfg:
#                 inputs1, inputs2, padded_samples = Mul_stack_batch(
#                     inputs1=inputs1,
#                     inputs2=inputs2,
#                     size=self.test_cfg.get('size', None),
#                     numtimes=self.numtimes,
#                     size_divisor=self.test_cfg.get('size_divisor', None),
#                     pad_val=self.pad_val,
#                     seg_pad_val=self.seg_pad_val)
#                 for data_sample, pad_info in zip(data_samples, padded_samples):
#                     data_sample.set_metainfo({**pad_info})
#             else:
#                 inputs1 = torch.stack(inputs1, dim=0)
#                 inputs2 = torch.stack(inputs2, dim=0)

#         return dict(inputs1=inputs1, inputs2=inputs2, data_samples=data_samples)
