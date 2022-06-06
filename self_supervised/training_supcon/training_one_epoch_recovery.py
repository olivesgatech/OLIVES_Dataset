from utils.utils import AverageMeter,warmup_learning_rate
import time
import torch
import sys

def train_Recovery(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (images, patient, week, weight, gender, diabetes_type, diabetes_years,hbalc,hyper,systolic_bp,diastolic_bp,smoking,age,bcva,cst) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.to(device)

        bsz = week.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method1 == 'patient':
            labels1 = patient.cuda()
        elif opt.method1 == 'bcva':
            labels1 = bcva.cuda()
        elif opt.method1 == 'cst':
            labels1 = cst.cuda()
        elif opt.method1 == 'age':
            labels1 = age.cuda()
        elif opt.method1 == 'gender':
            labels1 = gender.cuda()
        elif opt.method1 == 'diabetes_type':
            labels1 = diabetes_type.cuda()
        elif opt.method1 == 'diabetes_years':
            labels1 = diabetes_years.cuda()
        elif opt.method1 == 'hyper':
            labels1 = hyper.cuda()
        elif opt.method1 == 'hbalc':
            labels1 = hbalc.cuda()
        elif opt.method1 == 'systolic_bp':
            labels1 = systolic_bp.cuda()
        elif opt.method1 == 'diastolic_bp':
            labels1 = diastolic_bp.cuda()
        elif opt.method1 == 'smoking':
            labels1 = smoking.cuda()

        else:
            labels1 = 'Null'
        # Method 2
        if opt.method2 == 'patient':
            labels2 = patient.cuda()
        elif opt.method2 == 'bcva':
            labels2 = bcva.cuda()
        elif opt.method2 == 'cst':
            labels2 = cst.cuda()

        else:
            labels2 = 'Null'
        # Method 3
        if opt.method3 == 'patient':
            labels3 = patient.cuda()
        elif opt.method3 == 'bcva':
            labels3 = bcva.cuda()
        elif opt.method3 == 'cst':
            labels3 = cst.cuda()

        else:
            labels3 = 'Null'
        # Method 4
        if opt.method4 == 'patient':
            labels4 = patient.cuda()
        elif opt.method4 == 'bcva':
            labels4 = bcva.cuda()
        elif opt.method4 == 'cst':
            labels4 = cst.cuda()
        else:
            labels4 = 'Null'
        # Method 5
        if opt.method5 == 'patient':
            labels5 = patient.cuda()
        elif opt.method5 == 'bcva':
            labels5 = bcva.cuda()
        elif opt.method5 == 'cst':
            labels5 = cst.cuda()
        else:
            labels5 = 'Null'

        if (opt.num_methods == 0):
            loss = criterion(features)
        elif (opt.num_methods == 1):
            # labels1 = labels1.to(device)
            if (torch.isnan(bcva).any()):
                print(bcva)

            loss = criterion(features, labels1)

        elif (opt.num_methods == 2):
            loss = criterion(features, labels1) + criterion(features, labels2)
        elif (opt.num_methods == 3):
            loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features, labels3)
        elif (opt.num_methods == 4):
            loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features,
                                                                                           labels3) + criterion(
                features, labels4)
        elif (opt.num_methods == 5):
            loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features,
                                                                                           labels3) + criterion(
                features, labels4) + criterion(features, labels5)
        else:
            loss = 'Null'

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
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg