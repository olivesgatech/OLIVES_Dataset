import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import random
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from dataset import LoaderRCT, get_RCT
from datetime import datetime
import torchvision.models as torchmodels
from torch.utils.data import DataLoader
from oct_and_biomarker_model import MLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='RCT', help='Dataset')
parser.add_argument('--epochs', type=int, default=50, help='# of Epochs')
parser.add_argument('--resume', action='store', type=str,
                    default=None,
                    help='Resume checkpoint directory')
parser.add_argument('--seed', type=int, default=4567, help='Random seed') #4567 #2318
parser.add_argument('--save_dir', type=str, default='./runs/RCT/Resnet18', help='Folder to save checkpoints/logs')
parser.add_argument('--data_path', default='/home/yash-yee/projects/lab/Data/ZhangData/OCT/',
                    help='Folder for the datasets')
parser.add_argument('--patience', nargs='?', type=int, default=10,
                    help='Allows early stopping')
parser.add_argument('--eval', type=str, default='Patient',
                    choices=['Patient', 'Traditional'], help='Evaluate test set at patient level')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for \
                                training (default: auto)')
parser.add_argument('--which_baseline', type=str, default='img_att', choices=['img_only', 'att_only', 'img_att'])
parser.add_argument('--which_bio_analysis', type=str, default=None)
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
args = parser.parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def mainStandard():
    args.cuda = "cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    device = args.cuda
    print(f'Using device: {device}')

    start_time = time.time()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    current_time = datetime.now().strftime('%b%d_%H%M%S')

    if args.which_baseline == 'img_att':
        if args.which_bio_analysis == None:
            save_path = os.path.join(args.save_dir, 'Att_Guided', 'RCT_Train_Att_Guided_Bio_All_' +
                                     current_time)
        else:
            save_path = os.path.join(args.save_dir, 'Att_Guided', 'RCT_TrainFilt_Att_Guided_Bio_' +
                                     args.which_bio_analysis + '_' +
                                     current_time)
    elif args.which_baseline == 'img_only':
        save_path = os.path.join(args.save_dir, 'Baseline', 'RCT_Baseline_' + current_time)
    elif args.which_baseline == 'att_only':
        save_path = os.path.join(args.save_dir, 'AttBaseline', 'RCT_AttBaseline_' + current_time)
    print('Model will be saved here: ' + save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path, 0o777)
    log_dir = os.path.join(save_path, 'logs')

    data_configs = get_RCT(eval=args.eval, which_bio=args.which_bio_analysis, which_baseline=args.which_baseline)
    train_transform = transforms.Compose([])

    if args.which_baseline == 'att_only':
        mean = 0.18857142857142858
        std = 0.1998690464681048
    else:
        mean = 0.1690965894072076
        std = 0.07541925364584133


    train_transform.transforms.append(transforms.Grayscale(num_output_channels=3))
    train_transform.transforms.append(transforms.ToTensor())

    train_transform.transforms.append(transforms.Normalize(mean=mean, std=std))
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_dataset = LoaderRCT(data_dict=data_configs, transform=train_transform, split='tr')
    val_dataset = LoaderRCT(data_dict=data_configs, transform=val_transform, split='val')

    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.which_baseline == 'img_only':
        model1 = torchmodels.resnet18(pretrained=False)
        model1.fc = nn.Linear(512, 2)
        model1 = torch.nn.DataParallel(model1, device_ids=range(1))
        model1 = model1.to(device)  # Send to GPU
        optimizer1 = optim.Adam(model1.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    elif args.which_baseline == 'img_att':
        model1 = torchmodels.resnet18(pretrained=False)
        model1.fc = nn.Linear(512, 2)
        model1 = torch.nn.DataParallel(model1, device_ids=range(1))
        model1 = model1.to(device)  # Send to GPU
        optimizer1 = optim.Adam(model1.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        model2 = MLP(21, 2)
        model2 = torch.nn.DataParallel(model2, device_ids=range(1))
        model2 = model2.to(device)  # Send to GPU
        optimizer2 = optim.Adam(model2.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    elif args.which_baseline == 'att_only':
        model2 = MLP(21, 2)
        model2 = torch.nn.DataParallel(model2, device_ids=range(1))
        model2 = model2.to(device)  # Send to GPU
        optimizer2 = optim.Adam(model2.parameters(), lr=0.15, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True) # 0.3


    best_loss = 1e9
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.which_baseline == 'img_only':
                model1.load_state_dict(checkpoint['state_dict'])
            elif args.which_baseline == 'img_att':
                model1.load_state_dict(checkpoint['state_dict'])
                model2.load_state_dict(checkpoint['mlp_state_dict'])
            elif args.which_baseline == 'att_only':
                model2.load_state_dict(checkpoint['mlp_state_dict'])
            start_epoch = int(checkpoint['epoch'] + 1)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch + 1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()


    loss1 = torch.nn.BCEWithLogitsLoss()


    att_model = model2 if not args.which_baseline == 'img_only' else None
    att_opt = optimizer2 if not args.which_baseline == 'img_only' else None
    im_model = model1 if not args.which_baseline == 'att_only' else None
    im_opt = optimizer1 if not args.which_baseline == 'att_only' else None

    logger.info("*** Start training classifier ***")
    patience = 0
    for epoch in range(start_epoch, args.epochs):

        # Train
        _, _, _, _, _, _, _, _ = participants_classification(trainLoader, im_model, im_opt, epoch, loss1,
                                                             att_model=att_model, att_opt=att_opt)

        # Validate
        val_loss, val_acc = validateStandard(valLoader, im_model, epoch, loss1, att_model)

        if patience == args.patience:
            print('Early stopping')
            break

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            print(f"Best validation loss = {best_loss:.4f}")
            logger.info("*** Saving the BEST Classifier ***")
            output_checkpoint = os.path.join(save_path, "pytorch_ckpt_latest.tar")
            if args.which_baseline == 'img_att':
                torch.save(
                    {
                        "state_dict": model1.state_dict(),
                        "mlp_state_dict": att_model.state_dict(),
                        "optimizer_state_dict": optimizer1.state_dict(),
                        "mlp_optimizer_state_dict": att_opt.state_dict(),
                        "epoch": epoch
                    },
                    output_checkpoint,
                )
            elif args.which_baseline == 'img_only':
                torch.save(
                    {
                        "state_dict": model1.state_dict(),
                        "optimizer_state_dict": optimizer1.state_dict(),
                        "epoch": epoch
                    },
                    output_checkpoint,
                )
            elif args.which_baseline == 'att_only':
                torch.save(
                    {
                        "state_dict": att_model.state_dict(),
                        "optimizer_state_dict": att_opt.state_dict(),
                        "epoch": epoch
                    },
                    output_checkpoint,
                )
            print('Model is saved here: ' + save_path)
            # for early-stopping
            # the current loss is not getting smaller
            patience = 0
        else:
            patience += 1

        print(f"Patience = {patience}\n")

    print('Total processing time: %.4f seconds' % (time.time() - start_time))
    print('Model is saved here: ' + save_path)
    return save_path


def validateStandard(val_loader, model1, epoch, lossfn1, model2):

    if args.which_baseline == 'att_only':
        model2.eval()
    elif args.which_baseline == 'img_only':
        model1.eval()
    else:
        model1.eval()
        model2.eval()

    lossfn2 = nn.MSELoss()

    avg_loss_collect = torch.zeros([1, ]).cuda()
    avg_acc_collect = torch.zeros([1, ]).cuda()
    with torch.no_grad():
        for batch_idx, (feat, label, _, att) in enumerate(val_loader):
            if args.which_baseline == 'att_only':
                model2.zero_grad()
            elif args.which_baseline == 'img_only':
                model1.zero_grad()
            else:
                model1.zero_grad()
                model2.zero_grad()
            # model1.zero_grad()
            # model2.zero_grad()

            if args.which_baseline == 'att_only':
                feat, label = feat.to(DEVICE), label.to(DEVICE)
            else:
                feat, label, att = feat.to(model1.src_device_obj), label.to(model1.src_device_obj), att.to(model1.src_device_obj)
            one_hot = torch.zeros(label.shape[0], val_loader.dataset.nclasses)
            one_hot[range(label.shape[0]), label.long()] = 1
            one_hot = one_hot.cuda()

            if args.which_baseline == 'att_only':
                output2 = model2(feat.float())
                loss = lossfn1(output2, one_hot)
                pred_softmax2 = torch.log_softmax(output2, dim=1)
                _, preds2 = torch.max(pred_softmax2, dim=1)
            elif args.which_baseline == 'img_only':
                output1 = model1(feat.float())
                loss = lossfn1(output1, one_hot)
                pred_softmax1 = torch.log_softmax(output1, dim=1)
                _, preds1 = torch.max(pred_softmax1, dim=1)
            else:
                output1 = model1(feat.float())
                output2 = model2(att.float())
                loss1 = lossfn1(output1, one_hot)  #+ lossfn2(output1, output2)
                loss2 = lossfn1(output2, one_hot)
                pred_softmax1 = torch.log_softmax(output1, dim=1)
                _, preds1 = torch.max(pred_softmax1, dim=1)
                pred_softmax2 = torch.log_softmax(output2, dim=1)
                _, preds2 = torch.max(pred_softmax2, dim=1)
                loss3 = lossfn2(output1[preds2 == label], output2[preds2 == label])
                if torch.isnan(loss3):
                    loss = loss1 + loss2
                else:
                    loss = loss1 + loss2 + loss3

            avg_loss_collect += (loss * feat.shape[0])

    avg_loss_collect = avg_loss_collect / len(val_loader.dataset)

    logger.info('[Val] [Epoch %d / %d] Loss %.3f' % (epoch + 1, args.epochs, avg_loss_collect[0]))
    return avg_loss_collect, avg_acc_collect


def testStandard(classifier):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_configs = get_RCT(eval=args.eval, which_bio=args.which_bio_analysis, which_baseline=args.which_baseline)
    if args.which_baseline == 'att_only':
        mean = 0.18857142857142858
        std = 0.1998690464681048
    else:
        mean = 0.1690965894072076
        std = 0.07541925364584133
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_dataset = LoaderRCT(data_dict=data_configs, transform=test_transform, split='te')

    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.which_baseline == 'att_only':
        model = MLP(21, 2)
    else:
        model = torchmodels.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 2)

    # if someone has mutilple gpus set device_ids=[0]
    model = torch.nn.DataParallel(model, device_ids=range(1))
    model = model.to(device)  # Send to GPU
    state_dict = torch.load(classifier + '/pytorch_ckpt_latest.tar')
    model.load_state_dict(state_dict['state_dict'])

    model.eval()

    all_preds = []
    all_preds1 = []
    all_preds2 = []
    all_labels = []
    with torch.no_grad():
        for i, (feat, label, idxs, id) in enumerate(testLoader):
            model.zero_grad()
            # model2.zero_grad()

            feat, label = feat.to(device), label.to(device)
            output1 = model(feat.float())
            all_labels.extend(label.cpu().numpy())

            pred_softmax1 = torch.log_softmax(output1, dim=1)
            _, preds1 = torch.max(pred_softmax1, dim=1)

            all_preds1.extend(preds1.cpu().numpy())


    return all_preds, all_preds1, all_preds2, all_labels


def participants_classification(loader, model, opt, epoch, lossfn, att_model=None,
                                att_opt=None):
    '''Trains the model in the given epoch. It uses the training loader to get the dataset and trains the model
    for one epoch'''
    # sets model into training mode -> important for dropout batchnorm. etc.
    if args.which_baseline == 'img_only':
        model.train()
    elif args.which_baseline == 'att_only':
        att_model.train()
    else:
        model.train()
        att_model.train()

    # initializes cool bar for visualization
    tbar = tqdm(loader)

    # init statistics parameters
    train_loss = 0.0
    correct_samples = 0
    total = 0

    preds_and_output = np.array([], dtype=np.int64).reshape(0, loader.dataset.possible_bioindicators + 4)

    input_images = torch.tensor([]).cuda()
    old_outputs = torch.tensor([]).cuda()
    prev_preds = torch.tensor([]).cuda()
    # iterate over all samples in each batch i
    for i, (image, target, idxs, id) in enumerate(tbar):
        one_hot = torch.zeros(target.shape[0], loader.dataset.nclasses)
        one_hot[range(target.shape[0]), target.long()] = 1
        # assign each image and target to GPU
        if args.cuda:
            image, target = image.to(DEVICE), target.to(DEVICE),
            one_hot = one_hot.cuda()
            if args.which_baseline == 'img_att':
                id = id.to(DEVICE)

        if args.which_baseline == 'img_only':
            opt.zero_grad()
        elif args.which_baseline == 'att_only':
            att_opt.zero_grad()
        else:
            opt.zero_grad()
            att_opt.zero_grad()

        # convert image to suitable dims
        if args.which_baseline == 'img_only':
            image = image.float()
            output = model(image)
            logit, pred = torch.max(output.data, 1)
            input_images = torch.cat((input_images, image), 0)
            old_outputs = torch.cat((old_outputs, output), 0)
            prev_preds = torch.cat((prev_preds, pred), 0)

        elif args.which_baseline == 'att_only':
            att_output = att_model(image.float())
            attpred_softmax = torch.log_softmax(att_output, dim=1)
            att_logit, att_pred = torch.max(attpred_softmax, 1)
        else:
            image = image.float()
            output = model(image)
            logit, pred = torch.max(output.data, 1)
            att_output = att_model(id.float())
            attpred_softmax = torch.log_softmax(att_output, dim=1)
            att_logit, att_pred = torch.max(attpred_softmax, 1)


        total += target.size(0)


        if not args.which_baseline == 'att_only':
            t1 = np.expand_dims(idxs.cpu().numpy(), axis=1)
            if args.dataset == 'RCT':
                t2 = id.cpu().numpy()
            else:
                t2 = np.expand_dims(id.cpu().numpy(), axis=1)
            t3 = np.expand_dims(pred.cpu().numpy(), axis=1)
            t4 = np.expand_dims(target.cpu().numpy(), axis=1)
            t5 = np.expand_dims(logit.cpu().numpy(), axis=1)
            collect_preds_and_output = np.concatenate([t1, t2, t3, t4, t5], axis=1)
            preds_and_output = np.vstack([preds_and_output, collect_preds_and_output])
        else:
            t1 = np.expand_dims(idxs.cpu().numpy(), axis=1)
            if args.dataset == 'RCT':
                t2 = image.cpu().numpy()
            else:
                t2 = np.expand_dims(image.cpu().numpy(), axis=1)
            t3 = np.expand_dims(att_pred.cpu().numpy(), axis=1)
            t4 = np.expand_dims(target.cpu().numpy(), axis=1)
            t5 = np.expand_dims(att_logit.detach().cpu().numpy(), axis=1)
            collect_preds_and_output = np.concatenate([t1, t2, t3, t4, t5], axis=1)
            preds_and_output = np.vstack([preds_and_output, collect_preds_and_output])

        if args.which_baseline == 'img_att':
            loss1 = lossfn(output, one_hot)
            loss2 = lossfn(att_output, one_hot)
            lossfn2 = nn.MSELoss()


            loss3 = lossfn2(output[att_pred == target], att_output[att_pred == target])
            if torch.isnan(loss3):
                loss = loss1 + loss2
            else:
                loss = loss1 + loss2 + loss3
        elif args.which_baseline == 'img_only':
            loss = lossfn(output, one_hot)
        elif args.which_baseline == 'att_only':
            loss = lossfn(att_output, one_hot)
        # perform backpropagation
        loss.backward()

        # update params with gradients
        if args.which_baseline == 'img_only':
            opt.step()
        elif args.which_baseline == 'att_only':
            att_opt.step()
        else:
            opt.step()
            att_opt.step()

        # extract loss value as float and add to train_loss
        train_loss += loss.item()

        # Do fun bar stuff
        tbar.set_description(loader.dataset.split + ' loss: %.3f' % (train_loss / (i + 1)))

        if args.which_baseline == 'img_att':
            correct_pred1 = pred.eq(target.data)
            correct_pred2 = att_pred.eq(target)
            correct_samples += (correct_pred1 | correct_pred2).cpu().sum()
        elif args.which_baseline == 'img_only':
            correct_samples += pred.eq(target.data).cpu().sum()
        elif args.which_baseline == 'att_only':
            correct_samples += att_pred.eq(target.data).cpu().sum()


    # calculate accuracy
    acc = 100.0 * correct_samples.item() / total
    print('[Epoch: %d, numImages: %5d]' % (epoch, i * args.batch_size + image.data.shape[0]))
    print('Train Loss: %.3f' % train_loss)
    print(loader.dataset.split + ' Accuracy: %.3f' % acc)


    store_model_preds = {}
    if args.dataset == 'RCT':
        store_model_preds['indx'] = preds_and_output[:, 0]
        store_model_preds['id'] = preds_and_output[:, 1:loader.dataset.possible_bioindicators + 4 - 3]
        store_model_preds['prediction'] = preds_and_output[:, loader.dataset.possible_bioindicators + 4 - 3]
        store_model_preds['ground truth'] = preds_and_output[:, loader.dataset.possible_bioindicators + 4 - 2]
        store_model_preds['logit'] = preds_and_output[:, loader.dataset.possible_bioindicators + 4 - 1]

    else:
        store_model_preds['indx'] = preds_and_output[:, 0]
        store_model_preds['id'] = preds_and_output[:, 1]
        store_model_preds['prediction'] = preds_and_output[:, 2]
        store_model_preds['ground truth'] = preds_and_output[:, 3]
        store_model_preds['logit'] = preds_and_output[:, 4]

    return acc, store_model_preds, model, opt, input_images, old_outputs, prev_preds, train_loss


def confMatrix(truth, preds, savePath):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(truth, preds)
    classNames = ['DR', 'DME']

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    plt.rcParams.update({'font.size': 24})
    label_font = {'size': '24'}  # Adjust to fit
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classNames, yticklabels=classNames,
           # title='Subject ' + str(whichSubject) + ' Confusion Matrix', # change this to be your own model name
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(savePath + '/confMat.png')
    print("\n\nConfusion Matrix is saved here: " + savePath)


def computeMetrics(outputs, labels, savePath):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

    confMatrix(labels, outputs, savePath)

    accuracy = accuracy_score(labels, outputs)
    balanced_accuracy = balanced_accuracy_score(labels, outputs)
    precision = precision_score(labels, outputs, average=None)
    recall = recall_score(labels, outputs, average=None)
    f1 = f1_score(labels, outputs, average=None)

    print(f"Accuracy = {accuracy}")
    print(f"Balanced Accuracy = {balanced_accuracy}")
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1}")


if __name__ == '__main__':
    # BestStandardClassifier = mainStandard()
    BestStandardClassifier = '../Checkpoints/RCT_Train_Att_Guided_Bio_All_Jun08_010203'  # img_att
    comboPreds, standardPreds, mlpPreds, standardLabels = testStandard(BestStandardClassifier)

    print('\n\n******** Computing Standard Performance *************')
    computeMetrics(standardPreds, standardLabels, savePath=BestStandardClassifier)

