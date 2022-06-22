import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os

class PrimeDatasetAttributes(data.Dataset):
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
        frame_num = self.df.iloc[idx,3]
        age = self.df.iloc[idx,6]
        gender = self.df.iloc[idx,7]
        race = self.df.iloc[idx,8]
        diabetes_type = self.df.iloc[idx,9]
        diabetes_years = self.df.iloc[idx,10]
        bmi = self.df.iloc[idx,11]
        bcva = self.df.iloc[idx,12]
        drss = self.df.iloc[idx,13]
        cst = self.df.iloc[idx,14]
        li = self.df.iloc[idx,15]

        return image, patient, week, frame_num, age, gender, race, diabetes_type, diabetes_years,bmi,bcva,drss,cst,li