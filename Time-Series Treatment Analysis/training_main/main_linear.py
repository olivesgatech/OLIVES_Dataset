from __future__ import print_function
from config.config_linear import parse_option
from training_one_epoch.training_one_epoch_supervised import main
from training_one_epoch.training_one_epoch_oct_supervised_3d import main_oct_3D
from training_one_epoch.training_one_epoch_rnn import main_oct_rnn
from training_one_epoch.training_one_epoch_3D_RNN import main_oct_3d_rnn
try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass







if __name__ == '__main__':
    opt = parse_option()

    if opt.dataset == 'Fundus_Treatment':
        main()
    elif opt.dataset == 'Fundus_Time_Series':
        main_oct_rnn()
    elif opt.dataset == 'OCT_3D_Time_Series':
        main_oct_3d_rnn()
    else:
        main_oct_3D()