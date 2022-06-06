import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd


import os
class recovery(data.Dataset):
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
        patient = self.df.iloc[idx,1]
        week = self.df.iloc[idx,2]
        weight = self.df.iloc[idx,5]
        gender = self.df.iloc[idx,6]
        diabetes_type = self.df.iloc[idx,7]
        diabetes_years = self.df.iloc[idx,8]
        hbalc = self.df.iloc[idx,9]
        hyper = self.df.iloc[idx,10]
        systolic_bp = self.df.iloc[idx,11]
        diastolic_bp = self.df.iloc[idx,12]
        smoking = self.df.iloc[idx,13]
        age = self.df.iloc[idx, 14]
        bcva = self.df.iloc[idx, 15]
        cst = self.df.iloc[idx,16]

        return image, patient, week, weight, gender, diabetes_type, diabetes_years,hbalc,hyper,systolic_bp,diastolic_bp,smoking,age,bcva,cst