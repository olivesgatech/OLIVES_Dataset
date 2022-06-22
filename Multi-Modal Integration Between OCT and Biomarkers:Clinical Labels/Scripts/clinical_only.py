import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from torch import optim
from torchvision import transforms
from dataset import LoaderRCTcourse, get_RCTcourse
from datetime import datetime
from torch.utils.data import DataLoader
from clinical_only_model import MLP

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='RCT', help='Dataset')
parser.add_argument('--epochs', type=int, default=30, help='# of Epochs')
parser.add_argument('--resume', action='store', type=str,
                    default=None,
                    help='Resume checkpoint directory')
parser.add_argument('--seed', type=int, default=28563, help='Random seed') #86952 #28563 #98765 (best) #99836
parser.add_argument('--save_dir', type=str, default='./runs/RCT/Resnet18', help='Folder to save checkpoints/logs')
parser.add_argument('--patience', nargs='?', type=int, default=20,
                    help='Allows early stopping')
parser.add_argument('--eval', type=str, default='Patient',
                    choices=['Patient', 'Traditional'], help='Evaluate test set at patient level')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for \
                                training (default: auto)')
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
    save_path = os.path.join(args.save_dir, 'CourseBaseline', 'RCT_CourseBaseline_' + current_time)
    print('Model will be saved here: ' + save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_configs = get_RCTcourse()
    train_transform = transforms.Compose([])

    mean = 0.2581297341430058
    std = 0.17599633505555923

    train_transform.transforms.append([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_dataset = LoaderRCTcourse(data_dict=data_configs, transform=train_transform, split='tr')
    val_dataset = LoaderRCTcourse(data_dict=data_configs, transform=val_transform, split='val')

    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model2 = MLP(2, 2)
    model2 = torch.nn.DataParallel(model2, device_ids=range(1))
    model = model2.to(device)  # Send to GPU
    optimizer = optim.Adam(model2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    best_loss = 1e9
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model2.load_state_dict(checkpoint['state_dict'])
            start_epoch = int(checkpoint['epoch'] + 1)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch + 1))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit()

    # weights = torch.FloatTensor(np.array([1181 / 220, 1])).to(device)
    loss1 = torch.nn.BCEWithLogitsLoss()

    logger.info("*** Start training classifier ***")
    patience = 0
    for epoch in range(start_epoch, args.epochs):

        # Train
        _, _ = trainStandard(model, trainLoader, optimizer, loss1, epoch)

        # Validate
        val_avg_loss, val_avg_acc = validateStandard(model, valLoader, loss1, epoch)

        if patience == args.patience:
            print('Early stopping')
            break

        if val_avg_loss.item() < best_loss:
            best_loss = val_avg_loss.item()
            print(f"Best validation loss = {best_loss:.4f}")
            logger.info("*** Saving the BEST Classifier ***")
            output_checkpoint = os.path.join(save_path, "pytorch_ckpt_latest.tar")

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch
                },
                output_checkpoint,
            )
            print('Model is saved here: ' + save_path)

            patience = 0
        else:
            patience += 1

        print(f"Patience = {patience}\n")

    print('Total processing time: %.4f seconds' % (time.time() - start_time))
    print('Model is saved here: ' + save_path)
    return save_path


def trainStandard(model1, train_loader, opt1, lossfn1, epoch):
    model1.train()

    avg_loss_collect = torch.zeros([1, ]).cuda()
    avg_acc_collect = torch.zeros([1, ]).cuda()
    for batch_idx, (feat, label, idx) in tqdm(enumerate(train_loader)):
        model1.zero_grad()
        opt1.zero_grad()

        feat, label = feat.to(model1.src_device_obj), label.to(model1.src_device_obj)
        one_hot = torch.zeros(label.shape[0], train_loader.dataset.nclasses)
        one_hot[range(label.shape[0]), label.long()] = 1
        one_hot = one_hot.cuda()

        output1 = model1(feat.float())
        loss = lossfn1(output1, one_hot)

        # compute metrics
        pred_softmax1 = torch.log_softmax(output1, dim=1)
        _, preds1 = torch.max(pred_softmax1, dim=1)
        correct_pred1 = (preds1 == label)
        correct_pred = correct_pred1
        acc = correct_pred.sum() / len(correct_pred)
        acc = torch.round(acc) * 100

        loss.backward()
        opt1.step()

        if batch_idx % 20 == 0:
            print("Epoch [%d/%d] training Loss: %.4f" % (epoch + 1, args.epochs, loss.item()))

        avg_loss_collect += (loss * feat.shape[0])
        avg_acc_collect += (acc * feat.shape[0])

    avg_loss_collect = avg_loss_collect / len(train_loader.dataset)
    avg_acc_collect = avg_acc_collect / len(train_loader.dataset)
    logger.info('[Train] [Epoch %d / %d] Loss %.3f' % (epoch + 1, args.epochs, avg_loss_collect))
    return avg_loss_collect, avg_acc_collect


def validateStandard(model1, val_loader, lossfn1, epoch):
    model1.eval()

    avg_loss_collect = torch.zeros([1, ]).cuda()
    avg_acc_collect = torch.zeros([1, ]).cuda()
    with torch.no_grad():
        for batch_idx, (feat, label, idx) in enumerate(val_loader):
            model1.zero_grad()

            feat, label = feat.to(model1.src_device_obj), label.to(model1.src_device_obj)
            one_hot = torch.zeros(label.shape[0], val_loader.dataset.nclasses)
            one_hot[range(label.shape[0]), label.long()] = 1
            one_hot = one_hot.cuda()

            output1 = model1(feat.float())
            loss = lossfn1(output1, one_hot)

            # compute metrics
            pred_softmax1 = torch.log_softmax(output1, dim=1)
            _, preds1 = torch.max(pred_softmax1, dim=1)

            correct_pred1 = (preds1 == label)
            correct_pred = correct_pred1
            acc = correct_pred.sum() / len(correct_pred)
            acc = torch.round(acc) * 100

            avg_loss_collect += (loss * feat.shape[0])
            avg_acc_collect += (acc * feat.shape[0])

    avg_loss_collect = avg_loss_collect / len(val_loader.dataset)
    avg_acc_collect = avg_acc_collect / len(val_loader.dataset)

    logger.info('[Val] [Epoch %d / %d] Loss %.3f' % (epoch + 1, args.epochs, avg_loss_collect[0]))
    return avg_loss_collect, avg_acc_collect


def testStandard(classifier):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_configs = get_RCTcourse()

    mean = 0.25560041262846883
    std = 0.17419438655713454
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    test_dataset = LoaderRCTcourse(data_dict=data_configs, transform=test_transform, split='te')

    testLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MLP(2, 2)

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
        for i, (feat, label, idx) in enumerate(testLoader):
            model.zero_grad()

            feat, label = feat.to(device), label.to(device)

            output1 = model(feat.float())
            all_labels.extend(label.cpu().numpy())

            pred_softmax1 = torch.log_softmax(output1, dim=1)
            _, preds1 = torch.max(pred_softmax1, dim=1)

            all_preds1.extend(preds1.cpu().numpy())

    return all_preds, all_preds1, all_preds2, all_labels


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
    BestStandardClassifier = '../Checkpoints/RCT_CourseBaseline_May31_034306'
    _, standardPreds, _, standardLabels = testStandard(BestStandardClassifier)
    print('\n\n******** Computing Standard Performance *************')
    computeMetrics(standardPreds, standardLabels, savePath=BestStandardClassifier)