from torchvision import transforms, datasets
from datasets.fisheye_dataset import FisheyeFordDataset
from datasets.oct_dataset import OCTDataset
from datasets.biomarker import BiomarkerDatasetAttributes
from utils.utils import TwoCropTransform
from datasets.prime import PrimeDatasetAttributes
from datasets.prime_trex_combined import CombinedDataset
from datasets.recovery import recovery
from datasets.trex import TREX

from datasets.oct_cluster import OCTDatasetCluster
import torch
from models.resnet import SupConResNet
from loss.loss import SupConLoss
import torch.backends.cudnn as cudnn
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass
import torch.nn as nn
def set_model_contrast(opt):


    model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion


def set_loader(opt):
    # construct data loader


    if opt.dataset == 'OCT':

        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'OCT_Cluster':

        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Prime' or opt.dataset == 'CombinedBio' or opt.dataset == 'CombinedBio_Modfied' or opt.dataset =='Prime_Compressed':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'TREX_DME' or opt.dataset == 'Prime_TREX_DME_Fixed' \
            or opt.dataset == 'Prime_TREX_Alpha' or opt.dataset == 'Prime_TREX_DME_Discrete' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'PrimeBio':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'Prime_Comb_Bio':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'Recovery' or opt.dataset == 'Recovery_Compressed':

        mean = (.1706)
        std = (.2112)
    elif opt.dataset == 'path':

        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)


    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])



    if opt.dataset =='OCT':
        csv_path_train = opt.train_csv_path
        data_path_train = opt.train_image_path
        train_dataset = OCTDataset(csv_path_train,data_path_train,transforms = TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/full_prime_train.csv'
        data_path_train = opt.train_image_path
        train_dataset = PrimeDatasetAttributes(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'CombinedBio' or opt.dataset == 'CombinedBio_Modfied':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/complete_biomarker_training.csv'
        data_path_train =  opt.train_image_path#'/data/Datasets/Prime_FULL_128'
        train_dataset = BiomarkerDatasetAttributes(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'TREX_DME':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/datasets_combined/trex_compressed.csv'
        data_path_train = opt.train_image_path
        train_dataset = TREX(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_TREX_DME_Fixed' or opt.dataset == 'Prime_TREX_Alpha' \
            or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/datasets_combined/prime_trex_compressed.csv'
        data_path_train = opt.train_image_path
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_TREX_DME_Discrete':
        csv_path_train = './final_csvs_' + str(opt.patient_split) + '/Discretized_Datasets/cuts_' + str(opt.discrete_level) + ".csv"
        data_path_train = opt.train_image_path
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'Prime_Compressed':
        csv_path_train = './final_csvs_' + str(opt.patient_split) +'/datasets_combined/prime_compressed.csv'
        data_path_train = opt.train_image_path
        train_dataset = CombinedDataset(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    elif opt.dataset == 'OCT_Cluster':
        csv_path_train = opt.train_csv_path
        data_path_train = opt.train_image_path
        train_dataset = OCTDatasetCluster(csv_path_train, data_path_train, transforms=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler,drop_last=True)

    return train_loader


def set_model(opt):

    model = SupConResNet(name=opt.model)

    criterion = SupConLoss(temperature=opt.temp,device=opt.device)
    device = opt.device
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if opt.parallel == 1:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            model = model.to(device)
            criterion = criterion.to(device)
        cudnn.benchmark = True

    return model, criterion