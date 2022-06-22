from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2 as cv
import torch
import os


def get_RCT(eval=None, pseudo=False, which_bio=None, which_baseline=None):
    file_path = '../data/RCT/'

    if pseudo:
        raw_tr = np.load(os.path.join(file_path, 'train_pseudo.npy'), allow_pickle=True)
    elif which_bio != None and which_baseline == 'att_only':
        raw_tr = np.load(os.path.join(file_path, 'Privacy', 'train_' + which_bio + '.npy'), allow_pickle=True)
    elif which_baseline == 'att_only':
        raw_tr = np.load(os.path.join(file_path, 'train_att_only.npy'), allow_pickle=True)
    else:
        raw_tr = np.load(os.path.join(file_path, 'train.npy'), allow_pickle=True)

    if which_bio != None and which_baseline == 'att_only':
        raw_te = np.load(os.path.join(file_path, 'Privacy', 'test_' + which_bio + '.npy'), allow_pickle=True)
        patients_te = open(os.path.join(file_path, 'Privacy', 'test_' + which_bio + '.txt'), 'r')
        patients_te = patients_te.readlines()
        patients_te = np.asarray(patients_te)
    elif which_baseline == 'att_only':
        raw_te = np.load(os.path.join(file_path, 'test_att_only.npy'), allow_pickle=True)
        patients_te = open(os.path.join(file_path, 'test_att_only.txt'), 'r')
        patients_te = patients_te.readlines()
        patients_te = np.asarray(patients_te)
    else:
        raw_te = np.load(os.path.join(file_path, 'test.npy'), allow_pickle=True)

    if which_baseline == 'att_only' and which_bio == None:
        raw_val = np.load(os.path.join(file_path, 'val_att_only.npy'), allow_pickle=True)
    else:
        raw_val = np.load(os.path.join(file_path, 'val.npy'), allow_pickle=True)

    # init output dict
    output_dict = {}
    if which_baseline == 'att_only':
        output_dict['att_only'] = True
        output_dict['X_tr_bioindicators'] = raw_tr[:, 3]
        output_dict['Y_tr'] = raw_tr[:, 1]
        output_dict['X_te_bioindicators'] = raw_te[:, 3]
        output_dict['Y_te'] = raw_te[:, 1]
        output_dict['X_val_bioindicators'] = raw_val[:, 3]
        output_dict['Y_val'] = raw_val[:, 1]
        output_dict['tr_len'] = len(output_dict['Y_tr'])
        output_dict['te_len'] = len(output_dict['Y_te'])
        output_dict['val_len'] = len(output_dict['Y_val'])
        # if which_bio != None and which_baseline == 'att_only':
        output_dict['te_patients'] = patients_te
    else:
        output_dict['att_only'] = False
        output_dict['X_tr'] = raw_tr[:, 0]
        output_dict['Y_tr'] = raw_tr[:, 1]
        output_dict['X_te'] = raw_te[:, 0]
        output_dict['Y_te'] = raw_te[:, 1]
        output_dict['X_val'] = raw_val[:, 0]
        output_dict['Y_val'] = raw_val[:, 1]
        output_dict['tr_len'] = len(output_dict['Y_tr'])
        output_dict['te_len'] = len(output_dict['Y_te'])
        output_dict['val_len'] = len(output_dict['Y_val'])
        output_dict['Y_te_ID'] = np.squeeze(raw_te[:, 2])  # patient ID in test set
        output_dict['Y_tr_ID'] = np.squeeze(raw_tr[:, 2])  # patient ID in train set
        output_dict['Y_val_ID'] = np.squeeze(raw_val[:, 2])  # patient ID in train set
        output_dict['X_tr_bioindicators'] = raw_tr[:, 3]  # list of bio-indicators present per slice in train set
        output_dict['X_val_bioindicators'] = raw_val[:, 3]  # list of bio-indicators present per slice in train set

    output_dict['eval'] = eval
    output_dict['nclasses'] = 2
    return output_dict


class LoaderRCT(Dataset):
    def __init__(self, data_dict, split='tr', current_idxs=None, transform=None):
        if split != 'tr' and split != 'te' and split != 'val':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')
        self.split = split
        self.current_idxs = current_idxs
        self.ID = None
        self.bio = None
        self.eval = data_dict['eval']
        self.ID_bio = None
        self.data_dict = data_dict

        if 'te_patients' in data_dict:
            self.patients = data_dict['te_patients']

        if data_dict['att_only']:
            self.X = data_dict['X_' + split + '_bioindicators']
            self.Y = data_dict['Y_' + split]
        else:
            self.X = data_dict['X_' + split]
            self.Y = data_dict['Y_' + split]
            if split == 'te':
                self.ID = np.squeeze(data_dict['Y_te_ID'][current_idxs])
            elif split == 'val':
                self.ID = np.squeeze(data_dict['Y_val_ID'][current_idxs])
                self.ID_bio = np.concatenate((data_dict['Y_val_ID'][current_idxs],
                                              data_dict['X_val_bioindicators'][current_idxs]), axis=0)
                self.bio = np.squeeze(data_dict['X_val_bioindicators'][current_idxs])
            else:
                self.ID_bio = np.concatenate((data_dict['Y_tr_ID'][current_idxs],
                                              data_dict['X_tr_bioindicators'][current_idxs]), axis=0)
                self.ID = np.squeeze(data_dict['Y_tr_ID'][current_idxs])
                self.bio = np.squeeze(data_dict['X_tr_bioindicators'][current_idxs])  # bio-indicators in train set

            self.transform = transform
        self.possible_bioindicators = 21
        self.nclasses = data_dict['nclasses']

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        # if self.split == 'te':
        #     if 'te_patients' in self.data_dict:
        #         if self.patients[index].startswith('TREX'):
        #             patient = self.patients[index].split('/')[2]
        #         else:
        #             patient = self.patients[index].split('/')[1]
        if self.data_dict['att_only']:
            x = torch.tensor(x)
            # if self.split == 'te':
            #     return x, y, index, patient
            # else:
            return x, y, index
        else:
            if self.split == 'te':
                ID = self.ID[index]
            else:
                ID = self.ID[index]
                # if self.bio[index] == '999':
                #     bio = [int(self.bio[index])]
                # else:
                bio = list(map(int, self.bio[index]))  # convert list of strings to list of int
                pad = [0]*(self.possible_bioindicators - len(bio))
                # bio.insert(0, ID)  # bio is a list of [patientID, bioindicators, padding]
                bio.extend(pad)
                ID = bio
                ID = torch.tensor(ID)
                # print(len(ID))

            if len(x.shape) == 3:  # happens a few times in test set
                x = cv.cvtColor(x, cv.COLOR_BGR2GRAY)
            if len(x.shape) != 3:
                x = np.stack((x,)*3, axis=0)

            if self.transform is not None:
                permuted = np.moveaxis(x, [0, 1, 2], [2, 0, 1])
                im = Image.fromarray(permuted)
                x = self.transform(im)

            if self.current_idxs is not None:
                out_index = self.current_idxs[index]
            else:
                out_index = index

            if self.eval == 'Patient':
                return x, y, out_index, ID
            else:
                return x, y, out_index

        # if self.data_dict['att_only']:
        #     return x, y, index

    def __len__(self):
        return len(self.X)


def get_RCTcourse():
    file_path = '../data/RCT/'

    # patient, bcva, cst, label
    raw_tr = np.load(os.path.join(file_path, 'train_course_unique.npy'), allow_pickle=True)
    raw_te = np.load(os.path.join(file_path, 'test_course_unique.npy'), allow_pickle=True)
    raw_val = np.load(os.path.join(file_path, 'val_course_unique.npy'), allow_pickle=True)

    # init output dict
    output_dict = {}
    # output_dict['X_tr'] = raw_tr[:, 0]
    output_dict['Y_tr'] = raw_tr[:, 3]
    # output_dict['X_te'] = raw_te[:, 0]
    output_dict['Y_te'] = raw_te[:, 3]
    # output_dict['X_val'] = raw_val[:, 0]
    output_dict['Y_val'] = raw_val[:, 3]
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['val_len'] = len(output_dict['Y_val'])
    output_dict['X_te_ID'] = np.expand_dims(raw_te[:, 0], axis=1)  # patient ID in test set
    output_dict['X_tr_ID'] = np.expand_dims(raw_tr[:, 0], axis=1) # patient ID in train set
    output_dict['X_val_ID'] = np.expand_dims(raw_val[:, 0], axis=1) # patient ID in train set
    output_dict['X_te_bcva'] = np.expand_dims(raw_te[:, 1], axis=1)  # patient ID in test set
    output_dict['X_tr_bcva'] = np.expand_dims(raw_tr[:, 1], axis=1)  # patient ID in train set
    output_dict['X_val_bcva'] = np.expand_dims(raw_val[:, 1], axis=1)  # patient ID in train set
    output_dict['X_te_cst'] = np.expand_dims(raw_te[:, 2], axis=1)  # patient ID in test set
    output_dict['X_tr_cst'] = np.expand_dims(raw_tr[:, 2], axis=1)  # patient ID in train set
    output_dict['X_val_cst'] = np.expand_dims(raw_val[:, 2], axis=1)  # patient ID in train set
    output_dict['nclasses'] = 2
    return output_dict


def get_RCTcoursefusion():
    file_path = '../data/RCT/'

    #image, label, patient, bcva, cst
    raw_tr = np.load(os.path.join(file_path, 'train_course.npy'), allow_pickle=True)
    raw_te = np.load(os.path.join(file_path, 'test_course.npy'), allow_pickle=True)
    raw_val = np.load(os.path.join(file_path, 'val_course.npy'), allow_pickle=True)

    # init output dict
    output_dict = {}
    output_dict['X_tr'] = raw_tr[:, 0]
    output_dict['Y_tr'] = raw_tr[:, 1]
    output_dict['X_te'] = raw_te[:, 0]
    output_dict['Y_te'] = raw_te[:, 1]
    output_dict['X_val'] = raw_val[:, 0]
    output_dict['Y_val'] = raw_val[:, 1]
    output_dict['tr_len'] = len(output_dict['Y_tr'])
    output_dict['te_len'] = len(output_dict['Y_te'])
    output_dict['val_len'] = len(output_dict['Y_val'])
    output_dict['X_te_ID'] = np.expand_dims(raw_te[:, 2], axis=1)  # patient ID in test set
    output_dict['X_tr_ID'] = np.expand_dims(raw_tr[:, 2], axis=1) # patient ID in train set
    output_dict['X_val_ID'] = np.expand_dims(raw_val[:, 2], axis=1) # patient ID in train set
    output_dict['X_te_bcva'] = np.expand_dims(raw_te[:, 3], axis=1)  # patient ID in test set
    output_dict['X_tr_bcva'] = np.expand_dims(raw_tr[:, 3], axis=1)  # patient ID in train set
    output_dict['X_val_bcva'] = np.expand_dims(raw_val[:, 3], axis=1)  # patient ID in train set
    output_dict['X_te_cst'] = np.expand_dims(raw_te[:, 4], axis=1)  # patient ID in test set
    output_dict['X_tr_cst'] = np.expand_dims(raw_tr[:, 4], axis=1)  # patient ID in train set
    output_dict['X_val_cst'] = np.expand_dims(raw_val[:, 4], axis=1)  # patient ID in train set
    output_dict['nclasses'] = 2
    return output_dict


class LoaderRCTcourse(Dataset):
    def __init__(self, data_dict, split='tr', transform=None):
        if split != 'tr' and split != 'te' and split != 'val':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')
        self.split = split
        self.data_dict = data_dict

        # self.X = np.concatenate((data_dict['X_' + split + '_ID'], data_dict['X_' + split + '_bcva'],
        #                          data_dict['X_' + split + '_cst']), axis=1)
        self.X = np.concatenate((data_dict['X_' + split + '_bcva'],
                                 data_dict['X_' + split + '_cst']), axis=1)
        self.Y = data_dict['Y_' + split]

        self.transform = transform
        self.nclasses = data_dict['nclasses']

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = torch.tensor(x.astype(np.float))
        return x, y, index

    def __len__(self):
        return len(self.X)


class LoaderRCTcoursefusion(Dataset):
    def __init__(self, data_dict, split='tr', transform=None):
        if split != 'tr' and split != 'te' and split != 'val':
            raise Exception('Dataset handling is not correct!!! Split name should be either \'tr\' or \'te\'!!!')
        self.split = split
        self.data_dict = data_dict

        self.X1 = data_dict['X_' + split]
        self.X2 = np.concatenate((data_dict['X_' + split + '_bcva'],
                                  data_dict['X_' + split + '_cst']), axis=1)
        self.Y = data_dict['Y_' + split]

        self.transform = transform
        self.nclasses = data_dict['nclasses']

    def __getitem__(self, index):
        x1, x2, y = self.X1[index], self.X2[index], self.Y[index]

        if len(x1.shape) == 3:
            x1 = cv.cvtColor(x1, cv.COLOR_BGR2GRAY)
        if len(x1.shape) != 3:
            x1 = np.stack((x1,)*3, axis=0)

        if self.transform is not None:
            permuted = np.moveaxis(x1, [0, 1, 2], [2, 0, 1])
            im = Image.fromarray(permuted)
            x1 = self.transform(im)

        x2 = torch.tensor(x2.astype(np.float))
        return x1, x2, y, index

    def __len__(self):
        return len(self.X1)
