from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import math


class WiSigDiffRecvDataset(Dataset):
    """
    Rx: part of ["1-1", "1-19", "2-1", "2-19", "3-19", "7-7", "7-14", "8-8", "14-7", "18-2", "19-2", "20-1"]
    2-19, 7-14, 19-2, 20-1
    """
    def __init__(self, path='/home/yl/Data/Wisig/Rx/', Rx: list[str] = None, transform=None, train=True, return_index=False):
        super().__init__()
        data, targets = [], []
        for r in Rx:
            dataset = np.load(Path(path) / r / ('train.npz' if train else 'test.npz'))
            data.append(dataset['data'])
            targets.append(dataset['targets'])
        self.data = torch.from_numpy(np.concatenate(data, 0)).float().transpose(1, 2)
        self.targets = torch.from_numpy(np.concatenate(targets, 0)).long()
        self.transform = transform
        self.return_index = return_index

        print(f"Successfully loaded dataset on Rx of {Rx}, with shape of {self.data.shape}, {self.targets.shape}")

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.return_index:
            return data, target, index
        else:
            return data, target

    def __len__(self):
        return len(self.targets)
    






    
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    def get_mean_std_value(data_loader):
        """
        It calculates the mean and standard deviation of the data in the given loader over **channels**.
        This function is designed for: 
            4D tensor of [batch_size, channels, height, width] or 3D of [batch_size, channels, length].
        Note that the num of samples should be divisible by batch_size (or mark drop_last=True)!

        :param loader: a DataLoader object that iterates over the dataset
        :return: mean and std
        """

        mean_sum, var_sum, num_batches = 0, 0, 0

        for data, _ in data_loader:
            dims = [0] + list(range(2, data.ndim))  # 总维度除了第一个维度 (channel)
            # 计算 batch 内除了维度 1 以外的均值和，dim=1 为通道数量，不用参与计算
            mean_sum += data.mean(dim = dims)
            # 计算 batch 内除了维度 1 以外的方差，dim=1 为通道数量，不用参与计算
            var_sum += data.var(dim = dims)
            # 统计batch的数量
            num_batches += 1
            
        mean = mean_sum / num_batches  
        std = torch.sqrt(var_sum/num_batches)  # 计算标准差
        
        return mean, std

    wisig = WiSigDiffRecvDataset(Rx=['14-7'], train=False)
    loader = DataLoader(wisig, batch_size = 100, shuffle=True)
    
    mean, std = get_mean_std_value(loader)
    print(f"mean={mean}, std={std}")
    print(wisig.targets.max())
    