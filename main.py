import os
import json
import sys

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


def run_test_classifier():
    print(f'Running SVM Classifier')
    svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')

    # Train the classifier
    train_data = np.random.rand(251, 180)
    test_data = np.random.rand(251, 180)
    train_labels = np.random.randint(2, size=251)
    test_labels = np.random.randint(2, size=251)

    svc.fit(train_data, train_labels)

    # Evaluate the classifier

    predictions = svc.predict(test_data)
    report = classification_report(test_labels, predictions)
    print(report)


def eval_kern_params(subject, kern_type):
    eeg_train = EEGSubjectDataset(subject, train=True)

    # Create data loaders
    train_loader = DataLoader(eeg_train, batch_size=64, shuffle=True)

    out_path = f'./results//GridSearchCV//'

    old_stdout = sys.stdout
    log_file = open(f"{out_path}{subject}_{kern_type}_CV.log", "w")
    sys.stdout = log_file

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(kernel=kern_type, class_weight='balanced'), param_grid=param_grid, cv=cv, verbose=3)
    grid.fit(train_loader.dataset.data, train_loader.dataset.targets)

    sys.stdout = old_stdout
    log_file.close()

    out_path = f'./results//GridSearchCV//'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    result_str = "The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_)
    with open(f'{out_path}{subject}_{kern_type}_TRAIN_perf.txt', 'w') as handle:
        handle.write(result_str)

    return grid.best_params_, grid.best_score_


def eval_kernels(subject, kern_type, c_val, degree, gam):
    eeg_train = EEGSubjectDataset(subject, train=True)
    eeg_test = EEGSubjectDataset(subject, train=False)

    # Create data loaders
    train_loader = DataLoader(eeg_train, batch_size=64, shuffle=True)

    # Create the SVC classifier
    svc = SVC(kernel=kern_type, C=c_val, gamma=gam, degree=degree, class_weight='balanced')

    # Train the classifier
    svc.fit(train_loader.dataset.data, train_loader.dataset.targets)

    out_path = f'./results//{kern_type}//C_{c_val}_D{degree}//'

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Check the training accuracy (be sure the classifier is learning SOMETHING)
    train_preds = svc.predict(train_loader.dataset.data)
    train_report = classification_report(train_loader.dataset.targets, train_preds)

    with open(f'{out_path}{subject}_TRAIN_perf.txt', 'w') as handle:
        handle.write(train_report)


def run_all_CVs():
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\S_index.json'
    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    kern_types = ['linear', 'poly', 'rbf', 'sigmoid']
    best_score = 0.0
    best_params = None
    best_kern = None
    out_path = f'./results//GridSearchCV//'

    for entry in index:
        subject = entry['Subject']
        for kern_type in kern_types:
            params, score = eval_kern_params(subject, kern_type)
            if best_score < score < 1.0:
                best_score = score
                best_params = params
                best_kern = kern_type
        result_str = f'Kern: {best_kern}, Params: {best_params}, score: {best_score}'
        with open(f'{out_path}{subject}_CV.txt', 'w') as handle:
            handle.write(result_str)


def run_all_kernels():
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\S_index.json'
    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    kern_types = ['linear', 'poly', 'rbf', 'sigmoid']
    for entry in index:
        subject = entry['Subject']
        for kern_type in kern_types:
            eval_kernels(subject, kern_type)


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

    # Check the training accuracy (be sure the classifier is learning SOMETHING
    train_preds = svc.predict(train_loader.dataset.data)
    train_report = classification_report(train_loader.dataset.targets, train_preds)
    with open(f'./results_tuned//{subject}_TRAIN_perf.txt', 'w') as handle:
        handle.write(train_report)

    # Evaluate the classifier

    print("Saving results...")
    predictions = svc.predict(test_loader.dataset.data)
    report = classification_report(test_loader.dataset.targets, predictions)
    with open(f'./results_tuned//{subject}_TEST_perf.txt', 'w') as handle:
        handle.write(report)


def run_all_subjects(patho=False):
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    for entry in index:
        subject = entry['Subject']
        run_subject_classifier(subject, c=100.0, gam=1000.0)


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
    run_all_subjects(patho=True)
    # run_test_classifier()
    # run_all_kernels()
    # run_all_CVs()
    #eval_kernels("SM64", 'rbf', 100.0, 10, 1000.0)
