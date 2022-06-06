import torch
from utils.utils import AverageMeter,warmup_learning_rate,accuracy
import sys
import time
import numpy as np
from sklearn.metrics import precision_score,recall_score
from config.config_linear import parse_option
from utils.utils import set_loader_new, set_model, set_optimizer, adjust_learning_rate
from models.resnet import SupCEResNet_Fusion
def train_supervised_fusion(train_loader, model,criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()


    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()

    for idx, (image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,fusion_vector) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = image.to(device)
        fusion_vector = fusion_vector.long()
        fusion_vector = fusion_vector.to(device)
        if (opt.biomarker == 'vit_deb'):
            labels = vit_deb
        elif (opt.biomarker == 'ir_hrf'):
            labels = ir_hrf
        elif (opt.biomarker == 'full_vit'):
            labels = full_vit
        elif(opt.biomarker == 'partial_vit'):
            labels = partial_vit
        elif(opt.biomarker == 'drt'):
            labels = drt
        else:
            labels = fluid_irf
        labels = labels.long()

        labels = labels.to(device)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss


        output = model(images,fusion_vector)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        acc1= accuracy(output, labels, topk=(1,))


        top1.update(acc1[0].item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate_supervised_fusion(val_loader, model,criterion, opt):
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
        for idx, (image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,fusion_vector) in enumerate(val_loader):

            images = image.float().to(device)
            fusion_vector = fusion_vector.long()
            if (opt.biomarker == 'vit_deb'):
                labels = vit_deb
            elif (opt.biomarker == 'ir_hrf'):
                labels = ir_hrf
            elif (opt.biomarker == 'full_vit'):
                labels = full_vit
            elif (opt.biomarker == 'partial_vit'):
                labels = partial_vit
            elif (opt.biomarker == 'drt'):
                labels = drt
            else:
                labels = fluid_irf
            labels = labels.long()
            label_list.append(labels.detach().cpu().numpy())
            labels = labels.to(device)
            fusion_vector = fusion_vector.to(device)
            bsz = labels.shape[0]

            # forward
            output = model(images,fusion_vector)
            loss = criterion(output, labels)
            _, pred = output.topk(1, 1, True, True)

            out_list.append(pred.detach().cpu().numpy())
            # update metric
            losses.update(loss.item(), bsz)
            acc1= accuracy(output, labels, topk=(1,))
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    label_array = np.array(label_list)

    out_array = np.array(out_list)
    prec = precision_score(label_array.flatten(), out_array.flatten())
    rec = recall_score(label_array.flatten(), out_array.flatten())
    print('Precision = ' + str(precision_score(label_array.flatten(), out_array.flatten())))
    print('Recall = ' + str(recall_score(label_array.flatten(), out_array.flatten())))
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg,prec,rec

def main_supervised_fusion():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, test_loader = set_loader_new(opt)



    device = opt.device

    acc_list = []
    prec_list = []
    rec_list = []
    for i in range(0, 3):
    # training routine
        model = SupCEResNet_Fusion(name='resnet18',num_classes=2)
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        optimizer = set_optimizer(opt, model)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            print(optimizer)

            # train for one epoch
            time1 = time.time()
            loss, acc = train_supervised_fusion(train_loader, model, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
                epoch, time2 - time1, acc))


        loss, test_acc,prec, recall = validate_supervised_fusion(test_loader, model, criterion, opt)
        acc_list.append(test_acc)
        prec_list.append(prec)
        rec_list.append(recall)
    with open(opt.results_dir, "a") as file:
        # Writing data to a file
        file.write(opt.ckpt + '\n')
        file.write(opt.biomarker + '\n')
        file.write(opt.train_csv_path + '\n')
        file.write(opt.test_csv_path + '\n')
        file.write('Accuracy: ' + str(sum(acc_list)/3) + '\n')
        file.write('Precision: ' + str(sum(prec_list) / 3) + '\n')
        file.write('Recall: ' + str(sum(rec_list) / 3) + '\n')
        file.write('\n')
    print('Accuracy: ' + str(sum(acc_list) / 3))
    print('Precision: ' + str(sum(prec_list) / 3))
    print('Recall: ' + str(sum(rec_list) / 3))
    print('best accuracy: {:.2f}'.format(best_acc))