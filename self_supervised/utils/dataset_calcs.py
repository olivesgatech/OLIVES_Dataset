
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
RS = 123

from tqdm import tqdm

import torch
from torchvision import transforms



from datasets.prime_trex_combined import CombinedDataset
csv_path_train = '/final_csvs_1/datasets_combined/trex_dme_compressed.csv'
data_path_train ='/data/Datasets/Prime_FULL'
mean = (.1651)
std = (.2118)
normalize = normalize = transforms.Normalize(mean=mean, std=std)
trans = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,

    ])
train_dataset = CombinedDataset(csv_path_train,data_path_train,transforms= trans)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=7, pin_memory=True)

def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0
    for data,_,_,_,_ in tqdm(loader):
        channels_sum += torch.mean(data)
        channels_squared_sum += torch.mean(data**2)
        num_batches+=1
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**.5

    return mean,std
mean,std = get_mean_std(train_loader)
print(mean)
print(std)

for idx, (images, labels,_,_,_) in enumerate(tqdm(train_loader)):
    images = images.float().cuda()

    ret_image = images
    ret_image = ret_image.squeeze().detach().cpu().numpy()


    plt.imshow(ret_image)
    plt.show()
