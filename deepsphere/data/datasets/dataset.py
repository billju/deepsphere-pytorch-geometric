"""Datasets for reduced atmospheric river and tropical cyclone detection dataset.
"""

import itertools
import os
import shutil
import h5py
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import Dataset, Data, extract_zip
from deepsphere.utils.get_ico_coords import get_ico_coords
from torchvision.datasets.utils import download_url


# pylint: disable=C0330


class ARTCDataset(Dataset):
    """
    Dataset for reduced atmospheric river and tropical cyclone dataset.
    """
    url = 'http://island.me.berkeley.edu/ugscnn/data/climate_sphere_l5.zip'

    def __init__(self, root, indices=None, transform=None):
        """Initialization.

        Args:
            path (str): Path to the data or desired place the data will be downloaded to.
            indices (list): List of indices representing the subset of the data used for the current dataset.
            transform_data (:obj:`transform.Compose`): List of torchvision transforms for the data.
            transform_labels (:obj:`transform.Compose`): List of torchvision transforms for the labels.
            download (bool): Flag to decide if data should be downloaded or not.
        """
        super(ARTCDataset, self).__init__(root, transform, None, None)
        self.files = h5py.File(self.processed_paths[0], 'r')
        self.N = self.files['data'].shape[0]
        self.idxs = indices if indices is not None else list(range(self.N))

    @property
    def raw_file_names(self):
        return 'climate_sphere_l5.zip'

    def download(self):
        download_url(self.url, self.raw_dir)

    @property
    def processed_file_names(self):
        return 'data_5_all.h5'

    def process(self):
        extract_zip(self.raw_paths[0], self.raw_dir, log=True)
        path = os.path.join(self.raw_dir, 'data_5_all')

        files = [os.path.join(path, f) for f in os.listdir(path)]
        N = len(files)

        x = np.load(files[0])

        data_shape = x['data'].T.shape
        label_shape = x['labels'].T.shape

        with h5py.File(os.path.join(self.processed_paths[0]), 'w') as hf:
            dset = hf.create_dataset("data",
                                     shape=(N,) + data_shape,
                                     chunks=(1,) + data_shape,
                                     dtype='f4',
                                     compression='gzip')
            lset = hf.create_dataset("labels",
                                     shape=(N,) + label_shape,
                                     chunks=(1,) + label_shape,
                                     dtype='f4',
                                     compression='gzip')

            i = 0
            for f in tqdm(files):
                x = np.load(f)

                dset[i] = np.ascontiguousarray(x['data'].T)
                lset[i] = np.ascontiguousarray(x['labels'].T)
                i += 1

        shutil.rmtree(path)

    def len(self):
        return len(self.idxs)

    def get(self, idx):
        idx = self.idxs[idx]
        data, labels = self.files['data'][idx], self.files["labels"][idx]
        return Data(x=torch.FloatTensor(data),
                    y=torch.FloatTensor(labels))


class ARTCH5Dataset(Dataset):
    """
    Dataset for reduced atmospheric river and tropical cyclone dataset.
    """

    def __init__(self, h5_file, indices=None, transform=None):
        """Initialization.

        Args:
            path (str): Path to the data or desired place the data will be downloaded to.
            indices (list): List of indices representing the subset of the data used for the current dataset.
            transform_data (:obj:`transform.Compose`): List of torchvision transforms for the data.
            transform_labels (:obj:`transform.Compose`): List of torchvision transforms for the labels.
            download (bool): Flag to decide if data should be downloaded or not.
        """
        super(ARTCH5Dataset, self).__init__(None, transform, None, None)
        self.files = h5py.File(h5_file, 'r')
        self.N = self.files['data'].shape[0]
        self.idxs = indices if indices is not None else list(range(self.N))

    def len(self):
        return len(self.idxs)

    def get(self, idx):
        idx = self.idxs[idx]
        data, labels = self.files['data'][idx], self.files["labels"][idx]
        return Data(x=torch.FloatTensor(data),
                    y=torch.FloatTensor(labels))


class ARTCTemporaldataset(ARTCDataset):
    """Dataset for reduced ARTC dataset with temporality functionality.
    """

    def __init__(
            self,
            path,
            sequence_length,
            prediction_shift=0,
            indices=None,
            transform_image=None,
            transform_labels=None,
            transform_sample=None,
            download=False,
    ):
        """Initialization. Sort by run and sort each run by date and time. The samples at the tendo of each run are invalid and are removed.
        The list is then flattened. Self.allowed contains the list of all valid indices. Self.files contains all indices for the construction
        of samples.

        Args:
            path (str): Path to the data or desired place the data will be downloaded to.
            indices (list): List of indices representing the subset of the data used for the current dataset.
            transform_data (:obj:`transform.Compose`): List of torchvision transforms for the data.
            transform_labels (:obj:`transform.Compose`): List of torchvision transforms for the labels.
            download (bool): Flag to decide if data should be downloaded or not.
            temporality_length (int): The number of images used per sample.
        """
        super().__init__(path, indices, None, None, download)
        self.transform_image = transform_image
        self.transform_labels = transform_labels
        self.transform_sample = transform_sample
        self.sequence_length = sequence_length
        self.prediction_shift = prediction_shift
        sorted_by_run_and_date = [sorted(self.get_runs([i])) for i in [1, 2, 3, 4, 6]]
        self.allowed = list(itertools.chain(
            *[run[: -(self.sequence_length + self.prediction_shift)] for run in sorted_by_run_and_date]))
        self.files = list(itertools.chain(*sorted_by_run_and_date))

    def __len__(self):
        """Get length of dataset.

        Returns:
            int: Number of files contained in the dataset.
        """
        return len(self.allowed)

    def __getitem__(self, idx):
        """Get an item from the dataset.

        Args:
            idx (int): The index of the desired datapoint.

        Returns:
            obj, obj: The data and labels corresponding to the desired index. The type depends on the applied transforms.
        """
        sample = []
        idx = self.files.index(self.allowed[idx])
        for i in range(self.sequence_length):
            sample.append(np.load(os.path.join(self.path, self.files[idx + i])))
        data = [image["data"] for image in sample]
        if self.prediction_shift > 0:
            target = np.load(os.path.join(self.path, self.files[idx + i + self.prediction_shift]))
            labels = target["labels"]
        else:
            labels = sample[-1]["labels"]
        if self.transform_image:
            for i, image in enumerate(data):
                data[i] = self.transform_image(image)
        if self.transform_labels:
            labels = self.transform_labels(labels)
        if self.transform_sample:
            data = self.transform_sample(data)
        return data, labels


if __name__ == '__main__':
    from torch_geometric.data import DenseDataLoader

    dataset = ARTCDataset('/home/joey_yu/Datasets/data_5_all')
    loader = DenseDataLoader(dataset[:100], 4, shuffle=True, num_workers=0)

    print(len(dataset))
    for x in loader:
        print(x)
