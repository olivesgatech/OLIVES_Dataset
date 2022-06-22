import torch
from utils.utils import AverageMeter,warmup_learning_rate,accuracy
import sys
import time
import numpy as np
from config.config_linear import parse_option
from utils.utils import set_loader_new, set_model, set_optimizer, adjust_learning_rate, accuracy_multilabel
from sklearn.metrics import roc_auc_score
from models.resnet import SupCEResNet_Fusion

def train_supervised_multilabel_fusion(train_loader, model,criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, bio_tensor,clinical_tensor) in enumerate(train_loader):
        data_time.update(time.time() - end)
        labels = bio_tensor
        clinical_tensor = clinical_tensor.float()
        clinical_tensor = clinical_tensor.to(device)
        images = image.to(device)



        labels = labels.float()

        labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss


        output = model(images,clinical_tensor)

        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)


        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'.format(
                   epoch, idx + 1, len(train_loader)))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate_supervised_multilabel_fusion(val_loader, model,criterion, opt):
    """validation"""
    model.eval()


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    device = opt.device
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, bio_tensor,clinical_tensor) in (enumerate(val_loader)):

            images = image.float().to(device)
            labels = bio_tensor
            clinical_tensor = clinical_tensor.float()
            clinical_tensor = clinical_tensor.to(device)

            labels = labels.float()

            label_list.append(labels.squeeze().detach().cpu().numpy())
            labels = labels.to(device)

            bsz = labels.shape[0]

            # forward
            output = model(images,clinical_tensor)

            loss = criterion(output, labels)
            _, pred = output.topk(1, 1, True, True)

            out_list.append(output.detach().cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)


    label_array = np.concatenate(label_list,axis = 0)

    out_array = np.concatenate(out_list,axis=0)
    r = roc_auc_score(label_array,out_array,multi_class='ovr',average='weighted')

    out_array = np.array(out_list)

    return losses.avg, r

def main_supervised_multilabel_fusion():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader_new(opt)



    device = opt.device

    acc_list = []

    for i in range(0, 1):
    # training routine
        model = SupCEResNet_Fusion(name='resnet18',num_classes=5)
        model = model.to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        criterion = criterion.to(device)
        optimizer = set_optimizer(opt, model)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)


            # train for one epoch
            time1 = time.time()

            loss, acc = train_supervised_multilabel_fusion(train_loader, model, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))

            # eval for one epoch

        loss, test_acc = validate_supervised_multilabel_fusion(test_loader, model, criterion, opt)
        acc_list.append(test_acc)

    with open(opt.results_dir, "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(opt.biomarker + '\n')
        file.write(opt.train_csv_path + '\n')
        file.write('AUROC: ' + str(sum(acc_list)) + '\n')