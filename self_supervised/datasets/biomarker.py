import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os

class BiomarkerDatasetAttributes(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        atrophy = self.df.iloc[idx,2]
        EZ = self.df.iloc[idx,3]
        DRIL = self.df.iloc[idx,4]
        IR_hemm = self.df.iloc[idx,5]
        ir_hrf = self.df.iloc[idx,6]
        partial_vit = self.df.iloc[idx,7]
        full_vit = self.df.iloc[idx,8]
        preret_tiss = self.df.iloc[idx,9]
        vit_deb = self.df.iloc[idx,10]
        vmt = self.df.iloc[idx,11]
        drt = self.df.iloc[idx,12]
        fluid_irf = self.df.iloc[idx,13]
        fluid_srf = self.df.iloc[idx,14]

        rpe = self.df.iloc[idx,15]
        ga = self.df.iloc[idx,18]
        shrm = self.df.iloc[idx,19]
        eye_id = self.df.iloc[idx,22]
        bcva = self.df.iloc[idx,23]
        cst = self.df.iloc[idx,24]
        patient = self.df.iloc[idx,25]
        return image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient