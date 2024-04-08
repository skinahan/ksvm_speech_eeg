# data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn import preprocessing
from EEGSubjectDatasetLarge import EEGSubjectDatasetLarge


# Data Preprocessing Module

def set_seed(sd):
    global seed
    seed = sd


def get_seed():
    return seed

def set_c_POW(cp):
    global c_POW
    c_POW = cp

def get_c_POW():
    return c_POW

def train_test_split_helper(feat_type, subject):
    patho = False
    if subject.startswith('S'):
        patho = True

    full_dataset = EEGSubjectDatasetLarge(feat_type, subject, patho=patho)
    # filter to only unique samples?
    df_full = pd.DataFrame(full_dataset.data)

    # filtered_df = df_full.drop_duplicates()
    # perform train/test split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    y = full_dataset.labels
    set_seed = seed
    X_train, X_test, y_train, y_test = train_test_split(df_full, y, test_size=0.2, random_state=set_seed)

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    train_labels = y_train
    test_labels = y_test

    return X_train, X_test, train_labels, test_labels


# NOTE: DO NOT USE FOR HYPERPARAM TUNING
# For a specified subject:
# - Split the F1-F3 feature sets into train/test
# - Normalize (or Scale) the data
# - Perform Feature Selection via KPCA
# - Concatenate the resulting feature sets
def get_data(subj, scaleTransform):
    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_true_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subj)

    X2_train, X2_true_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subj)

    X3_train, X3_true_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subj)

    if scaleTransform:
        scaler1 = preprocessing.StandardScaler()
        scaler2 = preprocessing.StandardScaler()
        scaler3 = preprocessing.StandardScaler()

        X_train1 = scaler1.fit_transform(X1_train)
        X_train2 = scaler2.fit_transform(X2_train)
        X_train3 = scaler3.fit_transform(X3_train)

        X1_true_test = scaler1.transform(X1_true_test)
        X2_true_test = scaler2.transform(X2_true_test)
        X3_true_test = scaler3.transform(X3_true_test)

    norm = False
    # normalize data
    if norm:
        X_train1 = preprocessing.normalize(X1_train, axis=0)
        X_train2 = preprocessing.normalize(X2_train, axis=0)
        X_train3 = preprocessing.normalize(X3_train, axis=0)

        X1_true_test = preprocessing.normalize(X1_true_test, axis=0)
        X2_true_test = preprocessing.normalize(X2_true_test, axis=0)
        X3_true_test = preprocessing.normalize(X3_true_test, axis=0)

    # Perform feature selection / compression via KPCA
    kernel_pca1 = KernelPCA(
        n_components=60, kernel='poly'
    )
    kernel_pca2 = KernelPCA(
        n_components=60, kernel='poly'
    )
    kernel_pca3 = KernelPCA(
        n_components=60, kernel='poly'
    )
    kernel_pca1.fit(X_train1)
    kernel_pca2.fit(X_train2)
    kernel_pca3.fit(X_train3)

    # transform the training set (feature selection)
    X_train1 = kernel_pca1.transform(X_train1)
    X_train2 = kernel_pca2.transform(X_train2)
    X_train3 = kernel_pca3.transform(X_train3)

    # transform the test set (held out from feature selection)
    X_test1 = kernel_pca1.transform(X1_true_test)
    X_test2 = kernel_pca2.transform(X2_true_test)
    X_test3 = kernel_pca3.transform(X3_true_test)

    X_train = np.hstack([X_train1, X_train2, X_train3])
    X_test = np.hstack([X_test1, X_test2, X_test3])

    y_train = [train1_labels[i] for i, val in enumerate(train1_labels)]
    y_test = [test1_labels[i] for i, val in enumerate(test1_labels)]

    return (X_train, y_train), (X_test, y_test)


def normalize_combine_data(X1, X2, X3, scaleTransform):
    X_train1, X_valid1, X_test1 = X1
    X_train2, X_valid2, X_test2 = X2
    X_train3, X_valid3, X_test3 = X3

    if scaleTransform:
        scaler1 = preprocessing.StandardScaler()
        scaler2 = preprocessing.StandardScaler()
        scaler3 = preprocessing.StandardScaler()

        X_train1 = scaler1.fit_transform(X_train1)
        X_train2 = scaler2.fit_transform(X_train2)
        X_train3 = scaler3.fit_transform(X_train3)

        X_valid1 = scaler1.transform(X_valid1)
        X_valid2 = scaler2.transform(X_valid2)
        X_valid3 = scaler3.transform(X_valid3)

        X1_true_test = scaler1.transform(X_test1)
        X2_true_test = scaler2.transform(X_test2)
        X3_true_test = scaler3.transform(X_test3)

    # Perform feature selection / compression via KPCA
    # NOTE: fit is performed while EXCLUDING outer fold (X_test_) to prevent data leakage
    kernel_pca1 = KernelPCA(
        n_components=60, kernel='poly'
    )
    kernel_pca2 = KernelPCA(
        n_components=60, kernel='poly'
    )
    kernel_pca3 = KernelPCA(
        n_components=60, kernel='poly'
    )
    kernel_pca1.fit(X_train1)
    kernel_pca2.fit(X_train2)
    kernel_pca3.fit(X_train3)

    # transform the training set (feature selection)
    X_train1 = kernel_pca1.transform(X_train1)
    X_train2 = kernel_pca2.transform(X_train2)
    X_train3 = kernel_pca3.transform(X_train3)

    # transform the validation set (held out from feature selection)
    X_valid1 = kernel_pca1.transform(X_valid1)
    X_valid2 = kernel_pca2.transform(X_valid2)
    X_valid3 = kernel_pca3.transform(X_valid3)

    # transform the test set (held out from feature selection)
    X_test1 = kernel_pca1.transform(X1_true_test)
    X_test2 = kernel_pca2.transform(X2_true_test)
    X_test3 = kernel_pca3.transform(X3_true_test)

    X_train = np.hstack([X_train1, X_train2, X_train3])
    X_validate = np.hstack([X_valid1, X_valid2, X_valid3])
    X_test = np.hstack([X_test1, X_test2, X_test3])

    return X_train, X_validate, X_test
