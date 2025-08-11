import  torch
from torch.utils.data import Dataset
import os
from PIL import Image

class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.gt = torch.load(label_path, weights_only=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join(self.dir, '%d.png' % idx))
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.gt[idx]
        return img, label
