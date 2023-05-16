import csv
import json
import numpy as np
import torch

from eeg_dataset import get_target_labels_binary, get_device


class EEGSubjectDatasetLarge():
    def __init__(self, feat_num, subject=None, patho=False):
        device = get_device()
        if patho:
            first_char = 'S'
        else:
            first_char = 'C'
        self.feat_num = feat_num
        # Load the index
        f = open(f'{first_char}_index.json')
        self.index = json.load(f)['index']
        f.close()

        # Load the feature data
        if feat_num == 0:
            self.data = np.load(f'{first_char}_all_f1s.npy')
        if feat_num == 1:
            self.data = np.load(f'{first_char}_all_f2s.npy', mmap_mode='r+')
        if feat_num == 2:
            self.data = np.load(f'{first_char}_all_f3s.npy', mmap_mode='r+')

        # Load the labels data
        with open(f'{first_char}_labels.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            self.labels = [''.join(j) for j in reader]
        target_labels, category = get_target_labels_binary(self.labels)
        pure_target_cats = [category[j] for j in self.labels]

        self.targets = torch.tensor(pure_target_cats, dtype=torch.int64, device=device)
        self.labels = pure_target_cats

        self.range = None
        # Filter data for a single subject, if specified
        if subject is not None:
            for entry in self.index:
                entry_subj = entry['Subject']
                if entry_subj == subject:
                    self.range = entry['range']
                    self.data = self.data[self.range[0]:self.range[1]]
                    self.labels = self.labels[self.range[0]:self.range[1]]

    def __len__(self):
        # the size of the set is equal to the length of the vector
        return len(self.labels)

    def __getitem__(self, i):
        # the method returns a pair: given - label for the index number i
        # data = self.get_feature(i)
        data = self.data[i]
        return data, self.labels[i]
