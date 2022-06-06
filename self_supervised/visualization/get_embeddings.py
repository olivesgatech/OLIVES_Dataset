

from config.config_linear import parse_option
import numpy as np
import torch

from tqdm import tqdm
from sklearn.manifold import TSNE

import matplotlib.patheffects as PathEffects

from models.resnet import  SupConResNet
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import seaborn as sns

from datasets.biomarker import BiomarkerDatasetAttributes

import scipy.stats as stats

def set_model(opt):

    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()




    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']
    device = opt.device
    if torch.cuda.is_available():
        if opt.parallel == 0:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.to(device)

        criterion = criterion.to(device)
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model

def set_loader_new(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100' or opt.dataset == 'Ford' or opt.dataset == 'Ford_Region':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'OCT':
        mean = (.1904)
        std = (.2088)
    elif opt.dataset == 'Prime':
        mean = (.1706)
        std = (.2112)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([

        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize,
    ])




    data_path_train = opt.train_image_path
    csv_path_train = './final_csvs_' + str(opt.patient_split) +'/complete_biomarker_training.csv'
    if (opt.biomarker == 'vit_deb'):
        csv_path_test = './final_csvs_' + str(opt.patient_split) + '/test_biomarker_sets/test_VD.csv'
    elif (opt.biomarker == 'ir_hrf'):
        csv_path_test = './final_csvs_' + str(opt.patient_split) + '/test_biomarker_sets/test_IRHRF.csv'
    elif (opt.biomarker == 'full_vit'):
        csv_path_test = './final_csvs_' + str(opt.patient_split) +'/test_biomarker_sets/test_FAVF.csv'
    elif (opt.biomarker == 'partial_vit'):
        csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/test_biomarker_sets/test_PAVF.csv'
    elif (opt.biomarker == 'drt'):
        csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/test_biomarker_sets/test_DRT_ME.csv'
    elif(opt.multi == 1):
        csv_path_test = './final_csvs_'+ str(opt.patient_split) +'/complete_biomarker_test.csv'
    else:
        csv_path_test = './final_csvs_' + str(opt.patient_split) + '/test_biomarker_sets/test_fluirf.csv'

    data_path_test = opt.test_image_path
    train_dataset = BiomarkerDatasetAttributes(csv_path_train,data_path_train,transforms = train_transform)
    test_dataset = BiomarkerDatasetAttributes(csv_path_test,data_path_test,transforms = val_transform)


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    if(opt.biomarker == 'drt' and opt.patient_split == 1):
        dl = True
    elif(opt.multi == 1):
        dl = True
    else:
        dl=False
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=0, pin_memory=True,drop_last=dl)

    return train_loader, test_loader
def main():
    opt =parse_option()
    train_loader,test_loader = set_loader_new(opt)
    model = set_model(opt)
    y, matrix = get_embeddings(test_loader,model)


    tsne = TSNE(random_state=32,n_components=1).fit_transform(matrix)
    class_present = np.zeros((500,1))
    class_absent = np.zeros((500,1))

    j=0
    k=0
    for i in range(0,len(tsne)):

        if(y[i] == 1):
            class_present[j] = tsne[i]
            j=j+1
        if (y[i] == 0):
            class_absent[k] = tsne[i]
            k = k + 1

    tsne_abs = abs(class_present)
    mu = np.mean(class_present)
    sigma = np.std(class_present)

    x_pres = np.linspace(mu - 1 * sigma, mu + 1 * sigma, 1000)
    plt.plot(x_pres, stats.norm.pdf(x_pres, mu, sigma),label = 'Biomarker Present')

    tsne_abs_2 = abs(class_absent)
    mu = np.mean(class_absent)
    sigma = np.std(class_absent)
    x_abs = np.linspace(mu - 1 * sigma, mu + 1 * sigma, 1000)

    plt.plot(x_abs, stats.norm.pdf(x_abs, mu, sigma),label = 'Biomarker Absent')
    plt.legend()
    plt.title('PAVF TSNE Gaussian Seperability')
    plt.show()

def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))
    classes = ['Absent','Present' ]
    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')

    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.legend(handles = sc.legend_elements()[0], labels = classes)

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(classes[i]), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.show()
    return f, ax
def get_embeddings(val_loader,model):
    arr = []
    labels_vec = []
    opt = parse_option()
    device = opt.device
    with torch.no_grad():
        for idx, (images, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(tqdm(val_loader)):
            images = images.float().to(device)
            labels = partial_vit

            features = model(images)

            vec = features.squeeze().detach().cpu().numpy()

            arr.append(vec)
            labels_vec.append(labels.item())
    y = np.array(labels_vec)
    matrix = np.array(arr)
    return y, matrix


if __name__ == '__main__':
    main()