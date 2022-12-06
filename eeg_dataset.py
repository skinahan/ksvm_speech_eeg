import category_encoders
import numpy as np
import sklearn.preprocessing
import torch

from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader

import json
import csv


# Encoding Helper Methods

def get_target_labels_cons(labels):
    label_cats = np.unique(labels)
    target_labels = []
    category = {}
    for label in label_cats:
        low_label = label.lower()
        if not low_label == 'read':
            label_split = low_label.split('_')
            label_word = label_split[1]
            startsWithB = label_word.startswith('b')
            startsWithM = label_word.startswith('m')
            startsWithP = label_word.startswith('p')
            if startsWithB or startsWithM or startsWithP:
                target_labels.append(label)
                if startsWithB:
                    category[label] = 0
                if startsWithM:
                    category[label] = 1
                if startsWithP:
                    category[label] = 2
        else:
            target_labels.append(label)
            category[label] = 3
    return target_labels, category


def get_target_labels_vows(labels):
    label_cats = np.unique(labels)
    target_labels = []
    category = {}
    # A, E, I, O, U
    vow_cats = ['read', 'a', 'e', 'i', 'o', 'u']
    for label in label_cats:
        low_label = label.lower()
        if not low_label == 'read':
            label_split = low_label.split('_')
            label_word = label_split[1]
            label_vow = label_word[1]
        else:
            label_vow = low_label
        category[label] = vow_cats.index(label_vow)
        target_labels.append(label)
    return target_labels, category


def get_target_labels_binary(labels):
    label_cats = np.unique(labels)
    target_labels = []
    category = {}
    vow_cats = ['read', 'speak']
    for label in label_cats:
        low_label = label.lower()
        if low_label == 'read':
            label_bin = low_label
        else:
            label_bin = 'speak'
        category[label] = vow_cats.index(label_bin)
        target_labels.append(label)
    return target_labels, category


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
    # If not...
    else:
        device = torch.device("cpu")
    # Default to CPU for now...
    device = torch.device("cpu")
    return device


class EEGSubjectDataset(Dataset):
    def __init__(self, subject, train=True):
        device = get_device()

        self.add_feats = True

        # base_dir = '/content/drive/MyDrive'
        # base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
        base_dir = './data'

        if train:
            data_type = 'train'
        else:
            data_type = 'test'

        orig_feat_set = np.load(f'{base_dir}\\{subject}_{data_type}_compressed.py')
        f4_feat_set = np.load(f'{base_dir}\\{subject}_{data_type}_f4s.npy')
        flat_f4 = f4_feat_set.flatten()
        self.ext_feats = flat_f4
        grad_feat_set = np.load(f'{base_dir}\\{subject}_{data_type}_f5s.npy')
        self.ext_feats2 = grad_feat_set
        if self.add_feats:
            self.data = np.hstack([orig_feat_set, grad_feat_set])
        else:
            self.data = orig_feat_set
        self.labels = np.load(f'{base_dir}\\{subject}_{data_type}_labels.npy')
        self.labels = [''.join(j).replace(" ", "") for j in self.labels]
        # print(self.labels)
        multiclass = False

        if multiclass:
            # we want b, p, m words, and reading
            target_labels, category = get_target_labels_vows(self.labels)
            num_cats = len(target_labels)
        else:
            # we want reading vs. everything else
            target_labels, category = get_target_labels_binary(self.labels)

        filtered_indices = [i for i, x in enumerate(self.labels) if x in target_labels]
        filtered_data = [self.data[i] for i in filtered_indices]
        filtered_labels = [self.labels[i] for i in filtered_indices]
        pure_target_cats = [category[j] for j in filtered_labels]

        self.targets = torch.tensor(pure_target_cats, dtype=torch.int64, device=device)
        self.labels = self.targets
        self.data = torch.tensor(filtered_data, dtype=torch.float32, device=device)

    def __len__(self):
        # the size of the set is equal to the length of the vector
        return len(self.data)

    def __str__(self):
        # we combine both data structures to present them in the form of a single table
        return str(torch.cat((self.data, self.labels.unsqueeze(1)), 1))

    def __getitem__(self, i):
        # the method returns a pair: given - label for the index number i
        # data = self.data[i]
        if self.add_feats:
            data = np.hstack([self.data[i], self.ext_feats])
        else:
            data = self.data[i]
        return data, self.labels[i]
