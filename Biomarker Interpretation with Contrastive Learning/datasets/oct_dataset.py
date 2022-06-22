import torch.utils.data as data
import os
from PIL import Image
import pandas as pd

class OCTDataset(data.Dataset):
    def __init__(self, df, img_dir, transforms):

        self.df = pd.read_csv(df)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,
                                self.df.iloc[idx, 1], self.df.iloc[idx, 0])
        im = Image.open(img_path).convert("L")

        image = self.transforms(im)
        label = self.df.iloc[idx, 2]
        patient_label = self.df.iloc[idx,3]
        return image, label, patient_label
