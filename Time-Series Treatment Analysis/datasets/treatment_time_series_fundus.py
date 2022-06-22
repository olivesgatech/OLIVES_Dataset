import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import torch
import os
class Fundus_Time_Series(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx, 0]

        volume = np.load(path)
        print(volume.shape)
        #volume = np.expand_dims(volume, axis =1)
        volume = self.transforms(volume)
        volume = volume.permute(1,0,2)
        #volume =volume[0:1,:,:]
        
        volume = volume.unsqueeze(1)

        treatment_label = self.df.iloc[idx,5]
        return volume, treatment_label