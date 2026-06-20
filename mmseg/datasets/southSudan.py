# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import Callable, Dict, List, Optional, Sequence, Union
import os
import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

from mmseg.registry import DATASETS

from mmengine.fileio import join_path
from mmengine.utils import is_abs

@DATASETS.register_module()
class MulSouthsudanDataset(BaseSegDataset):
    METAINFO = dict(
        classes=(
            # 'no data',
            'sorghum',
            'maize',
            'rice',
            'groundnut'
        ),
        palette = [
            # [0, 0, 0],
            [255, 255, 0],
            [5, 73, 7],
            [255, 165, 0],
            [128, 96, 0],
        ]
    )
    def __init__(self,
                 split: str,
                 img_suffix='.npz',
                 seg_map_suffix='.npz',
                 reduce_zero_label=None,
                 data_prefix: dict = dict(img_p1_path='', img_p2_path='', seg_map_path=''),
                 **kwargs) -> None:
        self.split = split
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            data_prefix = data_prefix,
            **kwargs)


    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        with open(self.split, 'r') as file:
            idx_list = [line.strip() for line in file.readlines()]
        # idx_list = np.array(idx_list).astype(np.int32)
        img_p1_dir = self.data_prefix.get('img_p1_path', None)
        img_p2_dir = self.data_prefix.get('img_p2_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        # imgs1 = np.array(os.listdir(img_p1_dir))
        # imgs2 = np.array(os.listdir(img_p1_dir))

        for num in idx_list:
            img = "southsudan_" + num + ".npz"
            data_info = dict(img_p1_path=osp.join(img_p1_dir,img), img_p2_path=osp.join(img_p2_dir,img))
            if ann_dir is not None:
                seg_map = osp.join(ann_dir, img)
                data_info["seg_map_path"] = seg_map
            data_info['label_map'] = self.label_map
            data_info['reduce_zero_label'] = self.reduce_zero_label
            data_info['seg_fields'] = []
            data_list.append(data_info)
        return data_list

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
            self.ann_file = join_path(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
              ``self.serialize_data`` is True.
        """
        # if self._fully_initialized:
        #     return
        # load data information
        self.data_list = self.load_data_list()
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True
