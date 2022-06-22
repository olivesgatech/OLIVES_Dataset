from utils.utils import AverageMeter,warmup_learning_rate
import time
import torch
import sys

def train_Bio(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, (images, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(train_loader):
        data_time.update(time.time() - end)



        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            if(opt.parallel == 1):
                images = images.cuda(non_blocking=True)
            else:
                images = images.to(opt.device)

        bsz = vit_deb.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # Cuda Version
        if(opt.parallel == 0):
            if opt.method1 == 'vit_deb':
                labels1 = vit_deb.to(device)
            elif opt.method1 == 'ir_hrf':
                labels1 = ir_hrf.to(device)
            elif opt.method1 == 'full_vit':
                labels1 = full_vit.to(device)
            elif opt.method1 == 'partial_vit':
                labels1 = partial_vit.to(device)
            elif opt.method1 == 'fluid_irf':
                labels1 = fluid_irf.to(device)
            elif opt.method1 == 'drt':
                labels1 = drt.to(device)
            elif opt.method1 == 'patient':
                labels1 = patient.to(device)
            elif opt.method1 == 'bcva':
                labels1 = bcva.to(device)
            elif opt.method1 == 'cst':
                labels1 = cst.to(device)


            else:
                labels1 = 'Null'
            # Method 2
            if opt.method2 == 'vit_deb':
                labels2 = vit_deb.to(device)
            elif opt.method2 == 'ir_hrf':
                labels2 = ir_hrf.to(device)
            elif opt.method2 == 'full_vit':
                labels2 = full_vit.to(device)
            elif opt.method2 == 'partial_vit':
                labels2 = partial_vit.to(device)
            elif opt.method2 == 'fluid_irf':
                labels2 = fluid_irf.to(device)
            elif opt.method2 == 'drt':
                labels2 = drt.to(device)
            elif opt.method2 == 'patient':
                labels2 = patient.to(device)
            elif opt.method2 == 'bcva':
                labels2 = bcva.to(device)
            elif opt.method2 == 'cst':
                labels2 = cst.to(device)


            else:
                labels2 = 'Null'
            # Method 3
            if opt.method3 == 'vit_deb':
                labels3 = vit_deb.to(device)
            elif opt.method3 == 'ir_hrf':
                labels3 = ir_hrf.to(device)
            elif opt.method3 == 'full_vit':
                labels3 = full_vit.to(device)
            elif opt.method3 == 'partial_vit':
                labels3 = partial_vit.to(device)
            elif opt.method3 == 'fluid_irf':
                labels3 = fluid_irf.to(device)
            elif opt.method3 == 'drt':
                labels3 = drt.to(device)
            elif opt.method3 == 'patient':
                labels3 = patient.to(device)
            elif opt.method3 == 'bcva':
                labels3 = bcva.to(device)
            elif opt.method3 == 'cst':
                labels3 = cst.to(device)


            else:
                labels3 = 'Null'
            # Method 4
            if opt.method4 == 'vit_deb':
                labels4 = vit_deb.to(device)
            elif opt.method4 == 'ir_hrf':
                labels4 = ir_hrf.to(device)
            elif opt.method4 == 'full_vit':
                labels4 = full_vit.to(device)
            elif opt.method4 == 'partial_vit':
                labels4 = partial_vit.to(device)
            elif opt.method4 == 'fluid_irf':
                labels4 = fluid_irf.to(device)
            elif opt.method4 == 'drt':
                labels4 = drt.to(device)
            elif opt.method4 == 'patient':
                labels4 = patient.to(device)
            elif opt.method4 == 'bcva':
                labels4 = bcva.to(device)
            elif opt.method4 == 'cst':
                labels4 = cst.to(device)


            else:
                labels4 = 'Null'
            # Method 5
            if opt.method5 == 'vit_deb':
                labels5 = vit_deb.to(device)
            elif opt.method5 == 'ir_hrf':
                labels5 = ir_hrf.to(device)
            elif opt.method5 == 'full_vit':
                labels5 = full_vit.to(device)
            elif opt.method5 == 'partial_vit':
                labels5 = partial_vit.to(device)
            elif opt.method5 == 'fluid_irf':
                labels5 = fluid_irf.to(device)
            elif opt.method5 == 'drt':
                labels5 = drt.to(device)
            elif opt.method5 == 'patient':
                labels5 = patient.to(device)
            elif opt.method5 == 'bcva':
                labels5 = bcva.to(device)
            elif opt.method5 == 'cst':
                labels5 = cst.to(device)

            else:
                labels5 = 'Null'
        else:
            #Method 1
            if opt.method1 == 'vit_deb':
                labels1 = vit_deb.cuda(non_blocking=True)
            elif opt.method1 == 'ir_hrf':
                labels1 = ir_hrf.cuda(non_blocking=True)
            elif opt.method1 == 'full_vit':
                labels1 = full_vit.cuda(non_blocking=True)
            elif opt.method1 == 'partial_vit':
                labels1 = partial_vit.cuda(non_blocking=True)
            elif opt.method1 == 'fluid_irf':
                labels1 = fluid_irf.cuda(non_blocking=True)
            elif opt.method1 == 'drt':
                labels1 = drt.cuda(non_blocking=True)
            elif opt.method1 == 'patient':
                labels1 = patient.cuda(non_blocking=True)
            elif opt.method1 == 'bcva':
                labels1 = bcva.cuda(non_blocking=True)
            elif opt.method1 == 'cst':
                labels1 = cst.cuda(non_blocking=True)

            elif opt.method1 == 'eye_id':
                labels1 = eye_id.cuda(non_blocking=True)
            else:
                labels1 = 'Null'
            # Method 2
            if opt.method2 == 'vit_deb':
                labels2 = vit_deb.cuda(non_blocking=True)
            elif opt.method2 == 'ir_hrf':
                labels2 = ir_hrf.cuda(non_blocking=True)
            elif opt.method2 == 'full_vit':
                labels2 = full_vit.cuda(non_blocking=True)
            elif opt.method2 == 'partial_vit':
                labels2 = partial_vit.cuda(non_blocking=True)
            elif opt.method2 == 'fluid_irf':
                labels2 = fluid_irf.cuda(non_blocking=True)
            elif opt.method2 == 'drt':
                labels2 = drt.cuda(non_blocking=True)
            elif opt.method2 == 'patient':
                labels2 = patient.cuda(non_blocking=True)
            elif opt.method2 == 'bcva':
                labels2 = bcva.cuda(non_blocking=True)
            elif opt.method2 == 'cst':
                labels2 = cst.cuda(non_blocking=True)

            elif opt.method2 == 'eye_id':
                labels2 = eye_id.cuda(non_blocking=True)
            else:
                labels2 = 'Null'
            # Method 3
            if opt.method3 == 'vit_deb':
                labels3 = vit_deb.cuda(non_blocking=True)
            elif opt.method3 == 'ir_hrf':
                labels3 = ir_hrf.cuda(non_blocking=True)
            elif opt.method3 == 'full_vit':
                labels3 = full_vit.cuda(non_blocking=True)
            elif opt.method3 == 'partial_vit':
                labels3 = partial_vit.cuda(non_blocking=True)
            elif opt.method3 == 'fluid_irf':
                labels3 = fluid_irf.cuda(non_blocking=True)
            elif opt.method3 == 'drt':
                labels3 = drt.cuda(non_blocking=True)
            elif opt.method3 == 'patient':
                labels3 = patient.cuda(non_blocking=True)
            elif opt.method3 == 'bcva':
                labels3 = bcva.cuda(non_blocking=True)
            elif opt.method3 == 'cst':
                labels3 = cst.cuda(non_blocking=True)

            elif opt.method3== 'eye_id':
                labels3 = eye_id.cuda(non_blocking=True)
            else:
                labels3 = 'Null'
            # Method 4
            if opt.method4 == 'vit_deb':
                labels4 = vit_deb.cuda(non_blocking=True)
            elif opt.method4 == 'ir_hrf':
                labels4 = ir_hrf.cuda(non_blocking=True)
            elif opt.method4 == 'full_vit':
                labels4 = full_vit.cuda(non_blocking=True)
            elif opt.method4 == 'partial_vit':
                labels4 = partial_vit.cuda(non_blocking=True)
            elif opt.method4 == 'fluid_irf':
                labels4 = fluid_irf.cuda(non_blocking=True)
            elif opt.method4 == 'drt':
                labels4 = drt.cuda(non_blocking=True)
            elif opt.method4 == 'patient':
                labels4 = patient.cuda(non_blocking=True)
            elif opt.method4 == 'bcva':
                labels4 = bcva.cuda(non_blocking=True)
            elif opt.method4 == 'cst':
                labels4 = cst.cuda(non_blocking=True)

            elif opt.method4 == 'eye_id':
                labels4 = eye_id.cuda(non_blocking=True)
            else:
                labels4 = 'Null'
            # Method 5
            if opt.method5 == 'vit_deb':
                labels5 = vit_deb.cuda(non_blocking=True)
            elif opt.method5 == 'ir_hrf':
                labels5 = ir_hrf.cuda(non_blocking=True)
            elif opt.method5 == 'full_vit':
                labels5 = full_vit.cuda(non_blocking=True)
            elif opt.method5 == 'partial_vit':
                labels5 = partial_vit.cuda(non_blocking=True)
            elif opt.method5 == 'fluid_irf':
                labels5 = fluid_irf.cuda(non_blocking=True)
            elif opt.method5 == 'drt':
                labels5 = drt.cuda(non_blocking=True)
            elif opt.method5 == 'patient':
                labels5 = patient.cuda(non_blocking=True)
            elif opt.method5 == 'bcva':
                labels5 = bcva.cuda(non_blocking=True)
            elif opt.method5 == 'cst':
                labels5 = cst.cuda(non_blocking=True)

            elif opt.method5 == 'eye_id':
                labels5 = eye_id.cuda(non_blocking=True)
            else:
                labels5 = 'Null'

        if(opt.num_methods == 0):
            loss = criterion(features)
        elif(opt.num_methods==1):
            labels1 = labels1.to(device)
            loss = criterion(features,labels1)
        elif(opt.num_methods == 2):
            if(opt.method1 == 'SimCLR'):
                loss = criterion(features) + criterion(features,labels2)
            else:
                loss = criterion(features,labels1) + criterion(features, labels2)
        elif(opt.num_methods == 3):
            if (opt.method1 == 'SimCLR'):

                loss = criterion(features) + criterion(features, labels2) + criterion(features, labels3)
            else:
                loss = criterion(features,labels1) + criterion(features,labels2) + criterion(features,labels3)
            if(torch.isnan(full_vit).any()):
                print(labels1)
        elif (opt.num_methods == 4):
            if (opt.method1 == 'SimCLR'):
                loss = criterion(features) + criterion(features, labels2) + criterion(features, labels3) + criterion(features,labels4)
            else:
                loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features, labels3) + criterion(features,labels4)
        elif (opt.num_methods == 5):
            if(opt.method1 == 'SimCLR'):
                loss = criterion(features) + criterion(features, labels2) + criterion(features, labels3) + criterion(features,labels4) + criterion(features,labels5)
            else:
                loss = criterion(features, labels1) + criterion(features, labels2) + criterion(features, labels3) + criterion(features,labels4) + criterion(features,labels5)
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