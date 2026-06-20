# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .typing_utils import SampleList


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def stack_batch(inputs: List[torch.Tensor],
                data_samples: Optional[SampleList] = None,
                size: Optional[tuple] = None,
                size_divisor: Optional[int] = None,
                pad_val: Union[int, float] = 0,
                seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs, list), \
        f'Expected input type to be list, but got {type(inputs)}'
    assert len({tensor.ndim for tensor in inputs}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs]}'
    assert inputs[0].ndim == 3, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs = []
    padded_samples = []
    inputs_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs]
    max_size = np.stack(inputs_sizes).max(0)
    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max_size = (max_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i in range(len(inputs)):
        tensor = inputs[i]
        if size is not None:
            width = max(size[-1] - tensor.shape[-1], 0)
            height = max(size[-2] - tensor.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max_size[-1] - tensor.shape[-1], 0)
            height = max(max_size[-2] - tensor.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]

        # pad img
        pad_img = F.pad(tensor, padding_size, value=pad_val)
        padded_inputs.append(pad_img)
        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if 'gt_sem_seg' in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape
            if 'gt_depth_map' in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(
                    gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo({
                'img_shape': tensor.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))

    return torch.stack(padded_inputs, dim=0), padded_samples



# def Mul_stack_batch(inputs1: List[torch.Tensor],
#                     inputs2: List[torch.Tensor],
#                     numtimes=0,
#                     s2_cloud=None,
#                     data_samples: Optional[SampleList] = None,
#                     size: Optional[tuple] = None,
#                     size_divisor: Optional[int] = None,
#                     pad_val: Union[int, float] = 0,
#                     seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
#     """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
#     to the max shape use the right bottom padding mode.

#     Args:
#         inputs (List[Tensor]): The input multiple tensors. each is a
#             CHW 3D-tensor.
#         data_samples (list[:obj:`SegDataSample`]): The list of data samples.
#             It usually includes information such as `gt_sem_seg`.
#         size (tuple, optional): Fixed padding size.
#         size_divisor (int, optional): The divisor of padded size.
#         pad_val (int, float): The padding value. Defaults to 0
#         seg_pad_val (int, float): The padding value. Defaults to 255

#     Returns:
#        Tensor: The 4D-tensor.
#        List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
#     """
#     assert isinstance(inputs1, list), \
#         f'Expected input type to be list, but got {type(inputs1)}'
#     assert len({tensor.ndim for tensor in inputs1}) == 1, \
#         f'Expected the dimensions of all inputs must be the same, ' \
#         f'but got {[tensor.ndim for tensor in inputs1]}'
#     assert inputs1[0].ndim == 4, f'Expected tensor dimension to be 3, ' \
#         f'but got {inputs1[0].ndim}'
#     assert len({tensor.shape[0] for tensor in inputs1}) == 1, \
#         f'Expected the channels of all inputs must be the same, ' \
#         f'but got {[tensor.shape[0] for tensor in inputs1]}'

#     assert isinstance(inputs2, list), \
#         f'Expected input type to be list, but got {type(inputs2)}'
#     assert len({tensor.ndim for tensor in inputs2}) == 1, \
#         f'Expected the dimensions of all inputs must be the same, ' \
#         f'but got {[tensor.ndim for tensor in inputs2]}'
#     assert inputs2[0].ndim == 4, f'Expected tensor dimension to be 3, ' \
#                                 f'but got {inputs2[0].ndim}'
#     assert len({tensor.shape[0] for tensor in inputs2}) == 1, \
#         f'Expected the channels of all inputs must be the same, ' \
#         f'but got {[tensor.shape[0] for tensor in inputs2]}'

#     # only one of size and size_divisor should be valid
#     assert (size is not None) ^ (size_divisor is not None), \
#         'only one of size and size_divisor should be valid'

#     padded_inputs1 = []
#     padded_inputs2 = []
#     padded_samples = []
    
#     inputs1_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs1]
#     max1_size = np.stack(inputs1_sizes).max(0)
#     inputs2_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs2]
#     max2_size = np.stack(inputs2_sizes).max(0)

#     if size_divisor is not None and size_divisor > 1:
#         # the last two dims are H,W, both subject to divisibility requirement
#         max1_size = (max1_size +
#                     (size_divisor - 1)) // size_divisor * size_divisor
#         max2_size = (max2_size +
#                     (size_divisor - 1)) // size_divisor * size_divisor

#     for i, (elem1, elem2,elem3) in enumerate(zip(inputs1, inputs2,s2_cloud)):
#         tensor1 = inputs1[i]
#         tensor2 = inputs2[i]
#         s2_cloud_path = s2_cloud[i]
#         if numtimes != 0 and s2_cloud_path != None:
#             if tensor1.shape[-3] < numtimes:
#                 padding_size = [0, 0, 0, 0, 0, numtimes-tensor1.shape[-3]]
#                 tensor1 = F.pad(tensor1, padding_size, mode='reflect')
#             elif tensor1.shape[-3] > numtimes:
#                 timestamps = tensor1.shape[1]
#                 # 随机采样时间点索引
#                 samples = np.random.choice(timestamps, size=numtimes, replace=False)
#                 # 关键排序操作：保持时间顺序
#                 samples.sort()  # <--- 这里对采样结
#                 tensor1 = tensor1[:,samples,:,:]
#             if tensor2.shape[-3] < numtimes:
#                 padding_size = [0, 0, 0, 0, 0, numtimes-tensor2.shape[-3]]
#                 tensor2 = F.pad(tensor2, padding_size, mode='reflect')
#             elif tensor2.shape[-3] > numtimes:
#                 cloud_data = np.load(s2_cloud_path)
#                 cloud_mask = cloud_data['s2_cloudmask']  # 形状为 (H, W, T)
#                 remapped = np.zeros_like(cloud_mask)
#                 remapped[cloud_mask == 0] = 3  # 清晰
#                 remapped[cloud_mask == 2] = 2  # 阴影
#                 remapped[cloud_mask == 3] = 1  # 薄雾
#                 cloud_scores = np.mean(remapped, axis=(0, 1))
#                 selected_indices = np.argsort(cloud_scores)[-numtimes:]
#                 selected_indices.sort()
#                 tensor2 = tensor2[:, selected_indices, :, :]

#         if size is not None:
#             width = max(size[-1] - tensor1.shape[-1], 0)
#             height = max(size[-2] - tensor1.shape[-2], 0)
#             # (padding_left, padding_right, padding_top, padding_bottom)
#             padding_size = (0, width, 0, height)
#         elif size_divisor is not None:
#             width = max(max1_size[-1] - tensor1.shape[-1], 0)
#             height = max(max1_size[-2] - tensor1.shape[-2], 0)
#             padding_size = (0, width, 0, height)
#         else:
#             padding_size = [0, 0, 0, 0]
#         # pad img1
#         pad_img = F.pad(tensor1, padding_size, value=pad_val)
#         padded_inputs1.append(pad_img)

#         if size is not None:
#             width = max(size[-1] - tensor2.shape[-1], 0)
#             height = max(size[-2] - tensor2.shape[-2], 0)
#             # (padding_left, padding_right, padding_top, padding_bottom)
#             padding_size = (0, width, 0, height)
#         elif size_divisor is not None:
#             width = max(max2_size[-1] - tensor2.shape[-1], 0)
#             height = max(max2_size[-2] - tensor2.shape[-2], 0)
#             padding_size = (0, width, 0, height)
#         else:
#             padding_size = [0, 0, 0, 0]
#         # pad img2
#         pad_img = F.pad(tensor2, padding_size, value=pad_val)
#         padded_inputs2.append(pad_img)


#         # pad gt_sem_seg
#         if data_samples is not None:
#             data_sample = data_samples[i]
#             pad_shape = None
#             if 'gt_sem_seg' in data_sample:
#                 gt_sem_seg = data_sample.gt_sem_seg.data
#                 del data_sample.gt_sem_seg.data
#                 data_sample.gt_sem_seg.data = F.pad(
#                     gt_sem_seg, padding_size, value=seg_pad_val)
#                 pad_shape = data_sample.gt_sem_seg.shape
#             if 'gt_edge_map' in data_sample:
#                 gt_edge_map = data_sample.gt_edge_map.data
#                 del data_sample.gt_edge_map.data
#                 data_sample.gt_edge_map.data = F.pad(
#                     gt_edge_map, padding_size, value=seg_pad_val)
#                 pad_shape = data_sample.gt_edge_map.shape
#             if 'gt_depth_map' in data_sample:
#                 gt_depth_map = data_sample.gt_depth_map.data
#                 del data_sample.gt_depth_map.data
#                 data_sample.gt_depth_map.data = F.pad(
#                     gt_depth_map, padding_size, value=seg_pad_val)
#                 pad_shape = data_sample.gt_depth_map.shape
#             data_sample.set_metainfo({
#                 'img1_shape': tensor1.shape[-2:],
#                 'img2_shape': tensor2.shape[-2:],
#                 'pad_shape': pad_shape,
#                 'padding_size': padding_size
#             })
#             padded_samples.append(data_sample)
#         else:
#             padded_samples.append(
#                 dict(
#                     img_padding_size=padding_size,
#                     pad_shape=pad_img.shape[-2:]))


#     stacked_inputs1 = torch.stack(padded_inputs1, dim=0)
#     stacked_inputs2 = torch.stack(padded_inputs2, dim=0)

#     return stacked_inputs1, stacked_inputs2, padded_samples

def Mul_stack_batch(inputs1: List[torch.Tensor],
                    inputs2: List[torch.Tensor],
                    numtimes=0,
                    data_samples: Optional[SampleList] = None,
                    size: Optional[tuple] = None,
                    size_divisor: Optional[int] = None,
                    pad_val: Union[int, float] = 0,
                    seg_pad_val: Union[int, float] = 255) -> torch.Tensor:
    """Stack multiple inputs to form a batch and pad the images and gt_sem_segs
    to the max shape use the right bottom padding mode.

    Args:
        inputs (List[Tensor]): The input multiple tensors. each is a
            CHW 3D-tensor.
        data_samples (list[:obj:`SegDataSample`]): The list of data samples.
            It usually includes information such as `gt_sem_seg`.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (int, float): The padding value. Defaults to 0
        seg_pad_val (int, float): The padding value. Defaults to 255

    Returns:
       Tensor: The 4D-tensor.
       List[:obj:`SegDataSample`]: After the padding of the gt_seg_map.
    """
    assert isinstance(inputs1, list), \
        f'Expected input type to be list, but got {type(inputs1)}'
    assert len({tensor.ndim for tensor in inputs1}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs1]}'
    assert inputs1[0].ndim == 4, f'Expected tensor dimension to be 3, ' \
        f'but got {inputs1[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs1}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs1]}'

    assert isinstance(inputs2, list), \
        f'Expected input type to be list, but got {type(inputs2)}'
    assert len({tensor.ndim for tensor in inputs2}) == 1, \
        f'Expected the dimensions of all inputs must be the same, ' \
        f'but got {[tensor.ndim for tensor in inputs2]}'
    assert inputs2[0].ndim == 4, f'Expected tensor dimension to be 3, ' \
                                f'but got {inputs2[0].ndim}'
    assert len({tensor.shape[0] for tensor in inputs2}) == 1, \
        f'Expected the channels of all inputs must be the same, ' \
        f'but got {[tensor.shape[0] for tensor in inputs2]}'

    # only one of size and size_divisor should be valid
    assert (size is not None) ^ (size_divisor is not None), \
        'only one of size and size_divisor should be valid'

    padded_inputs1 = []
    padded_inputs2 = []
    padded_samples = []
    inputs1_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs1]
    max1_size = np.stack(inputs1_sizes).max(0)
    inputs2_sizes = [(img.shape[-2], img.shape[-1]) for img in inputs2]
    max2_size = np.stack(inputs2_sizes).max(0)

    if size_divisor is not None and size_divisor > 1:
        # the last two dims are H,W, both subject to divisibility requirement
        max1_size = (max1_size +
                    (size_divisor - 1)) // size_divisor * size_divisor
        max2_size = (max2_size +
                    (size_divisor - 1)) // size_divisor * size_divisor

    for i, (elem1, elem2) in enumerate(zip(inputs1, inputs2)):
        tensor1 = inputs1[i]
        tensor2 = inputs2[i]
        if numtimes != 0:
            if tensor1.shape[-3] < numtimes:
                padding_size = [0, 0, 0, 0, 0, numtimes-tensor1.shape[-3]]
                tensor1 = F.pad(tensor1, padding_size, mode='reflect')
                # tensor1 = F.pad(tensor1, padding_size, mode='replicate')
                # tensor1 = F.pad(tensor1, padding_size, mode='constant')
            elif tensor1.shape[-3] > numtimes:
                timestamps = tensor1.shape[1]
                # 随机采样时间点索引
                samples = np.random.choice(timestamps, size=numtimes, replace=False)
                # 关键排序操作：保持时间顺序
                samples.sort()  # <--- 这里对采样结
                tensor1 = tensor1[:,samples,:,:]
            if tensor2.shape[-3] < numtimes:
                padding_size = [0, 0, 0, 0, 0, numtimes-tensor2.shape[-3]]
                tensor2 = F.pad(tensor2, padding_size, mode='reflect')
                # tensor2 = F.pad(tensor2, padding_size, mode='replicate')
            elif tensor2.shape[-3] > numtimes:
                timestamps = tensor2.shape[1]
                # 随机采样时间点索引
                samples = np.random.choice(timestamps, size=numtimes, replace=False)
                # 关键排序操作：保持时间顺序
                samples.sort()  # <--- 这里对采样结
                tensor2 = tensor2[:,samples,:,:]
        if size is not None:
            width = max(size[-1] - tensor1.shape[-1], 0)
            height = max(size[-2] - tensor1.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max1_size[-1] - tensor1.shape[-1], 0)
            height = max(max1_size[-2] - tensor1.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]
        # pad img1
        pad_img = F.pad(tensor1, padding_size, value=pad_val)
        padded_inputs1.append(pad_img)

        if size is not None:
            width = max(size[-1] - tensor2.shape[-1], 0)
            height = max(size[-2] - tensor2.shape[-2], 0)
            # (padding_left, padding_right, padding_top, padding_bottom)
            padding_size = (0, width, 0, height)
        elif size_divisor is not None:
            width = max(max2_size[-1] - tensor2.shape[-1], 0)
            height = max(max2_size[-2] - tensor2.shape[-2], 0)
            padding_size = (0, width, 0, height)
        else:
            padding_size = [0, 0, 0, 0]
        # pad img2
        pad_img = F.pad(tensor2, padding_size, value=pad_val)
        padded_inputs2.append(pad_img)


        # pad gt_sem_seg
        if data_samples is not None:
            data_sample = data_samples[i]
            pad_shape = None
            if 'gt_sem_seg' in data_sample:
                gt_sem_seg = data_sample.gt_sem_seg.data
                del data_sample.gt_sem_seg.data
                data_sample.gt_sem_seg.data = F.pad(
                    gt_sem_seg, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_sem_seg.shape
            if 'gt_edge_map' in data_sample:
                gt_edge_map = data_sample.gt_edge_map.data
                del data_sample.gt_edge_map.data
                data_sample.gt_edge_map.data = F.pad(
                    gt_edge_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_edge_map.shape
            if 'gt_depth_map' in data_sample:
                gt_depth_map = data_sample.gt_depth_map.data
                del data_sample.gt_depth_map.data
                data_sample.gt_depth_map.data = F.pad(
                    gt_depth_map, padding_size, value=seg_pad_val)
                pad_shape = data_sample.gt_depth_map.shape
            data_sample.set_metainfo({
                'img1_shape': tensor1.shape[-2:],
                'img2_shape': tensor2.shape[-2:],
                'pad_shape': pad_shape,
                'padding_size': padding_size
            })
            padded_samples.append(data_sample)
        else:
            padded_samples.append(
                dict(
                    img_padding_size=padding_size,
                    pad_shape=pad_img.shape[-2:]))


    stacked_inputs1 = torch.stack(padded_inputs1, dim=0)
    stacked_inputs2 = torch.stack(padded_inputs2, dim=0)

    return stacked_inputs1, stacked_inputs2, padded_samples
