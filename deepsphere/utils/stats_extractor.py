"""Get Means and Standard deviations for all features of a dataset.
"""
import numpy as np
import torch
from torch_geometric.data import Dataset


def stats_extractor(dataset: Dataset):
    """Iterates over a dataset object
    It is iterated over so as to calculate the mean and standard deviation.

    Args:
        dataset (:obj:`torch.utils.data.dataloader`): dataset object to iterate over

    Returns:
        :obj:numpy.array, :obj:numpy.array : computed means and standard deviation
    """

    V, F = dataset[0].x.shape
    summing = torch.zeros(F)
    square_summing = torch.zeros(F)
    total = 0

    for item in dataset:
        summing += torch.mean(item.x, dim=0)
        total += 1

    means = summing / total

    for item in dataset:
        square_summing += torch.sum((item.x - means) ** 2, dim=0)

    stds = np.sqrt(square_summing / (total * V - 1))

    return means.numpy(), stds.numpy()
