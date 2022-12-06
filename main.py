import torch

from sklearn.metrics import classification_report
from eeg_dataset import EEGSubjectDataset
import category_encoders
import numpy as np

from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from torch.utils.data import Dataset, DataLoader

import json
import csv

from sklearn.metrics import classification_report


def run_subject_classifier(subject, c=None, gam=None):
    eeg_train = EEGSubjectDataset(subject, train=True)
    eeg_test = EEGSubjectDataset(subject, train=False)

    # Create data loaders
    train_loader = DataLoader(eeg_train, batch_size=64, shuffle=True)

    test_loader = DataLoader(eeg_test, batch_size=64, shuffle=False)

    # Create the SVC classifier

    if c is not None:
        print(f'Running SVM Classifier for: {subject}, C: {c}, gamma: {gam}')
        svc = SVC(kernel='rbf', C=c, gamma=gam, class_weight='balanced')
    else:
        print(f'Running SVM Classifier for: {subject}')
        svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')

    # Train the classifier

    svc.fit(train_loader.dataset.data, train_loader.dataset.targets)

    # Evaluate the classifier

    print("Test Report")
    predictions = svc.predict(test_loader.dataset.data)

    print(classification_report(test_loader.dataset.targets, predictions))


def run_all_subjects():
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    f = open(f'{base_dir}\\index.json')
    index = json.load(f)['index']
    f.close()
    for entry in index:
        subject = entry['Subject']

        run_subject_classifier(subject)


def run_CV_single(subject):
    eeg_train = EEGSubjectDataset(subject, train=True)
    eeg_test = EEGSubjectDataset(subject, train=False)

    # Create data loaders
    train_loader = DataLoader(eeg_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(eeg_test, batch_size=64, shuffle=True)

    C_range = np.logspace(-2, 10, 16)
    # print(len(C_range))

    gamma_range = np.logspace(-9, 3, 16)
    # print(len(gamma_range))
    param_grid = dict(gamma=gamma_range, C=C_range)

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid=param_grid, cv=cv)  # , verbose=10)
    grid.fit(train_loader.dataset.data, train_loader.dataset.targets)
    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )

    best_C = grid.best_params_['C']
    best_gamma = grid.best_params_['gamma']

    run_subject_classifier(subject, best_C, best_gamma)


def run_CV_all():
    f = open('/content/drive/MyDrive/index.json')
    index = json.load(f)['index']
    f.close()
    for entry in index:
        subject = entry['Subject']
        print(f'Running GridSearchCV for: {subject}')
        run_CV_single(subject)


if __name__ == '__main__':
    run_all_subjects()
