import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd

import os
class TREX(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(path).convert("L")

        image = np.array(image)

        image = Image.fromarray(image)
        image = self.transforms(image)

        bcva=self.df.iloc[idx,1]

        snellen = self.df.iloc[idx,2]
        cst = self.df.iloc[idx, 3]
        eye_id = self.df.iloc[idx,4]
        patient_id = self.df.iloc[idx,5]
        return image, bcva,snellen,cst,eye_id,patient_id