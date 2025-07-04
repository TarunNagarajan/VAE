import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import Tuple, Optional

# load_data
# _split_data
# __len__
# __getitem__ : sample from the dataset

class DSpritesDataset(Dataset):
    def __init__(self,
                 root_dir: str = './data',
                 split: str = 'train',
                 transform: Optional[callable] = None,
                 download: bool = True):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # load data
        self.data, self.labels = self._load_data()

        # split data
        self._split_data()
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        import os
        import urllib.request

        data_path = os.path.join(self.root_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5')

        if not os.path.exists(data_path):
            os.makedirs(self.root_dir, exist_ok = True)
            print("NO PATH FOUND. ")

            print("Downloading dSprites Dataset...")
            url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5'
            urllib.request.urlretrieve(url, data_path)

            print("Download Complete!")

        # load data
        with h5py.File(data_path, 'r') as f:
            images = f['imgs'][:]
            labels = f['latents_values'][:]

        images = images.astype(np.float32)

        return images, labels
    
    def _split_data(self):
        total_samples = len(self.data)

        # split ratios (training, validation, testing)
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        # index access limit
        train_end = int(train_ratio * total_samples)
        val_end = int((train_ratio + val_ratio) * total_samples)

        if self.split == 'train':
            self.data = self.data[:train_end]
            self.labels = self.labels[:train_end]
        elif self.split == 'val':
            self.data = self.data[:val_end]
            self.labels = self.labels[:val_end]
        elif self.split == 'test':
            self.data = self.data[val_end:]
            self.labels = self.labels[val_end:]
        else:
            raise ValueError(f"INVALID SPLIT: {self.split}")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the Dataset"""

        image = self.data[idx]
        label = self.labels[idx]

        # fix: add channel dimension
        image = image[np.newaxis, :] # (1, 64, 64)

        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)

        # transform 
        if self.transform:
            image = self.transform(image)

        return image, label
        
