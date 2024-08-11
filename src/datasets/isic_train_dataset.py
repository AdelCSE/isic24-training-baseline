import io
import sys
import h5py
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

sys.path.append('../..')
from src.utils import remove_black_border

class ISICTrainDataset(Dataset):
    def __init__(self, df, hdf5_file, transforms=None):
        self.hdf5 = h5py.File(hdf5_file, mode='r')
        self.positive_df = df[df['target'] == 1].reset_index()
        self.negative_df = df[df['target'] == 0].reset_index()
        self.positive_ids = self.positive_df['isic_id'].values
        self.negative_ids = self.negative_df['isic_id'].values
        self.positive_targets = self.positive_df['target'].values
        self.negative_targets = self.negative_df['target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.positive_df) * 2
    
    def __getitem__(self, idx):
        
        if random.random() >= 0.5:
            df = self.positive_df
            ids = self.positive_ids
            targets = self.positive_targets
        else:
            df = self.negative_df
            ids = self.negative_ids
            targets = self.negative_targets
        
        idx = idx % df.shape[0]
        id = ids[idx]
        target = targets[idx]
        img = np.array(Image.open(io.BytesIO(self.hdf5[id][()])))

        img = remove_black_border(img)

        if self.transforms:
            img = self.transforms(image=img)['image']

        return {'image': img, 'target': target}