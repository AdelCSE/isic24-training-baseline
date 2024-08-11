import io
import sys
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

sys.path.append('../..')
from src.utils import remove_black_border

class ISICDataset(Dataset):
    def __init__(self, df, hdf5_file, transforms=None):
        self.hdf5 = h5py.File(hdf5_file, mode='r')
        self.df = df
        self.ids = self.df['isic_id'].values
        self.targets = self.df['target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.ids[idx]
        target = self.targets[idx]
        img = np.array(Image.open(io.BytesIO(self.hdf5[id][()])))

        img = remove_black_border(img)

        if self.transforms:
            img = self.transforms(image=img)['image']

        return {'image': img, 'target': target}