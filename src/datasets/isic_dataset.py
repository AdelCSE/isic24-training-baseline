import io
import sys
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional

sys.path.append('../..')
from src.utils import remove_black_border

class ISICDataset(Dataset):
    def __init__(self, 
                 df : str | pd.DataFrame, 
                 hdf5_file : str | h5py.File, 
                 transforms : Optional[Callable] = None
                 ) -> None:
        
        super().__init__()
        
        self.hdf5 = h5py.File(hdf5_file, mode='r')
        self.metadata = df
        self.transforms = transforms

    def __len__(self):
        return self.metadata.shape[0]

    def __getitem__(self, index):

        # Get indexed row
        row = self.metadata.iloc[index]

        # Get the target
        target = row['target']
        id = row['isic_id']

        # Get the image
        image_id = row['isic_id']
        img = np.array(Image.open(io.BytesIO(self.hdf5[image_id ][()])))
        
        # Remove black border in the image
        img = remove_black_border(img)

        # Apply image transformations
        if self.transforms:
            img = self.transforms(image=img)['image']

        return {'isic_id': id, 'image': img, 'target': target}