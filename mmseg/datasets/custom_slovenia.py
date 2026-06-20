import os
import scipy.io as scio
import numpy as np
from mmengine.dataset import Compose

class MultiModalDataset:
    def __init__(self, root_dir, split='train',pipeline1=None,pipeline2=None,test_mode=False):
        """
        初始化多模态数据集类。

        Args:
            root_dir (str): 数据集的根目录。
            split (str): 数据集的划分（'train', 'val', 'test'）。
        """
        self.root_dir = root_dir
        self.split = split

        # 构建数据集的文件路径
        self.label_dir = os.path.join(root_dir, 'label',split)
        self.s1_dir = os.path.join(root_dir, 's1', split)
        self.s2_dir = os.path.join(root_dir, 's2', split)

        # 加载所有文件
        self.samples = self._load_samples()
        self.pipeline1 = Compose(pipeline1)
        self.pipeline2 = Compose(pipeline2)
        self.test_mode = test_mode
    def _load_samples(self):
        """
        加载指定划分的数据集样本。

        Returns:
            list of dict: 每个字典包含'label', 's1_path', 's2_path'键。
        """
        samples = []

        # 遍历label文件夹下的所有_label.mat文件
        for filename in os.listdir(self.label_dir):
            if filename.endswith('_label.mat'):
                # 去掉文件名中的'_label.mat'，以便与s1和s2文件夹中的文件匹配
                base_name = filename[:-10]  # 假设后缀长度为9（'_label.mat'的长度）
                label_path = os.path.join(self.label_dir, filename)
                s1_path = os.path.join(self.s1_dir, f"{base_name}.mat")  # 假设s1文件没有特定后缀
                s2_path = os.path.join(self.s2_dir, f"{base_name}.mat")  # 假设s2文件也没有特定后缀

                # 检查文件是否存在
                if os.path.exists(label_path) and os.path.exists(s1_path) and os.path.exists(s2_path):
                    sample = {
                        'label': scio.loadmat(label_path)['label'],  # 假设'label'是存储标签的键
                        's1_path': s1_path,
                        's2_path': s2_path
                    }
                    samples.append(sample)
                if not os.path.exists(label_path):
                    raise FileNotFoundError(f"Label file not found: {label_path}")
                if not os.path.exists(s1_path):
                    raise FileNotFoundError(f"Sentinel-1 file not found: {s1_path}")
                if not os.path.exists(s2_path):
                    raise FileNotFoundError(f"Sentinel-2 file not found: {s2_path}")

        return samples

    def prepare_sample(self, index):
        """
        准备单个样本，包括加载图像和标签（如果不在测试模式下）。

        Args:
            index (int): 样本索引。

        Returns:
            dict: 包含处理后的图像和（可选的）标签的数据字典。
        """
        sample = self.samples[index]

        # 加载图像数据
        s1_data = self._load_image(sample['s1_path'])
        s2_data = self._load_image(sample['s2_path'])

        # 将数据封装成字典格式，以便传递给pipeline
        data_dict = {'s1_data': s1_data, 's2_data': s2_data}

        # 应用预处理（如果有的话）
        if self.pipeline1:
            data_dict['s1_data'] = self.pipeline1(data_dict['s1_data'])
        if self.pipeline2:
            data_dict['s2_data'] = self.pipeline2(data_dict['s2_data'])

            # 准备返回的数据字典
        result = {'img_sar': data_dict['s1_data'], 'img_optical': data_dict['s2_data']}
        if not self.test_mode:
            result['label'] = sample['label']

        return result

    def _load_image(self, path):
        """
        从给定路径加载图像数据。

        Args:
            path (str): 图像文件的路径。

        Returns:
            numpy.ndarray: 加载的图像数据。
        """
        # 这里我们简单地使用scipy.io.loadmat来加载.mat文件，但通常您会使用图像加载库（如PIL或OpenCV）
        # 对于.mat文件，我们假设它包含名为'band'的键，该键存储图像数据
        data = scio.loadmat(path)['band']
        # 可能需要额外的步骤来处理数据格式（如转置、缩放等）
        return data

    def __getitem__(self, index):
        """
        获取单个样本。

        Args:
            index (int): 样本索引。

        Returns:
            tuple: 包含标签、Sentinel-1图像数据和Sentinel-2图像数据的元组。
        """
        return self.prepare_sample(index)

    def __len__(self):
        """
        返回数据集中的样本数。

        Returns:
            int: 数据集样本总数。
        """
        return len(self.samples)

# 示例用法
dataset = MultiModalDataset(r'D:\openmm\SL-MULNet\data\slovenia', split='train')
dataset.prepare_sample(0)
# print(dataset[1])
print(f'Number of samples: {len(dataset)}')
label, s1_data, s2_data = dataset.samples[0]  # 获取第一个样本
# print(label.shape)
# print(s1_data.shape)
# print(s2_data.shape)