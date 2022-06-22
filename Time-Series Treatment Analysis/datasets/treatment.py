import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os
class OCT_Treatment(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(path)#.convert("L")
        #image = Image.fromarray(image)
        image = self.transforms(image)
        treatment_label = self.df.iloc[idx,6]
        return image, treatment_label