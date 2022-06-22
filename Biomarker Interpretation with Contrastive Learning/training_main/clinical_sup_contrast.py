from config.config_supcon import parse_option
from utils.utils_supcon import set_loader,set_model_contrast
from utils.utils import set_optimizer, adjust_learning_rate,save_model
import os
import time
import tensorboard_logger as tb_logger
from training_supcon.training_one_epoch_trex import train_TREX
from training_supcon.training_one_epoch_bio import train_Bio

from training_supcon.training_one_epoch_oct import train_OCT
from training_supcon.training_one_epoch_alpha import train_Alpha

from training_supcon.training_one_epoch_prime import train_Prime
from training_supcon.training_one_epoch_prime_trex_combined import train_Combined

def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model_contrast(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()

        if(opt.dataset =='CombinedBio' or opt.dataset =='CombinedBio_Modfied'):
            loss = train_Bio(train_loader, model, criterion, optimizer, epoch, opt)
        elif(opt.dataset == 'Prime'):
            loss = train_Prime(train_loader, model, criterion, optimizer, epoch, opt)
        elif(opt.dataset == 'TREX_DME'):
            loss = train_TREX(train_loader, model, criterion, optimizer, epoch, opt)
        elif(opt.dataset == 'Prime_TREX_DME_Fixed' or opt.dataset == 'Prime_TREX_DME_Discrete') \
                or opt.dataset == 'Patient_Split_2_Prime_TREX' or opt.dataset == 'Patient_Split_3_Prime_TREX':
            loss = train_Combined(train_loader, model, criterion, optimizer, epoch, opt)
        elif(opt.dataset == 'Prime_TREX_Alpha'):
            loss = train_Alpha(train_loader, model, criterion, optimizer, epoch, opt)
        elif (opt.dataset == 'OCT'):
            loss = train_OCT(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('loss', loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()