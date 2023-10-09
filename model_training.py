# model_training.py
import torch
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report

import json
import pandas as pd
import numpy as np
from data_preprocessing import *
from data_analysis import *

# Model Training and Evaluation Module


def nested_CV_Fixed_Gamma(subject, df, gamma):
    scaleTransform = True

    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subject)

    # assert train1_labels.dataset == train2_labels.dataset
    # assert train2_labels.dataset == train3_labels.dataset

    # configure the outer loop
    cv_outer = KFold(n_splits=4, shuffle=True, random_state=1)
    # define search space
    lower_C = -5
    upper_C = 5
    C_range = np.logspace(lower_C, upper_C, 32)
    # enumerate splits
    fold_num = 1
    for train_ix, test_ix in cv_outer.split(X1_train):
        train_ix = train_ix.astype(dtype=int).tolist()
        test_ix = test_ix.astype(dtype=int).tolist()

        assert train_ix != test_ix

        y_train = torch.utils.data.Subset(train1_labels, train_ix)
        y_validate = torch.utils.data.Subset(train1_labels, test_ix)

        # split data
        X_train1, X_valid1 = X1_train[train_ix, :], X1_train[test_ix, :]
        X_train2, X_valid2 = X2_train[train_ix, :], X2_train[test_ix, :]
        X_train3, X_valid3 = X3_train[train_ix, :], X3_train[test_ix, :]

        X1 = X_train1, X_valid1, X1_test
        X2 = X_train2, X_valid2, X2_test
        X3 = X_train3, X_valid3, X3_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        classifiers = []
        for C in C_range:
            # Create a model
            clf = SVC(kernel='rbf', class_weight='balanced', C=C, gamma=gamma)
            # Train the model
            clf.fit(X_train, y_train)

            yTrain = [y_train[i] for i, val in enumerate(y_train)]
            # Training accuracy
            trainHat = clf.predict(X_train)
            trainAcc = accuracy_score(yTrain, trainHat)

            # Evaluate the model
            y_valid = [y_validate[i] for i, val in enumerate(y_validate)]
            yhat = clf.predict(X_validate)
            acc = accuracy_score(y_valid, yhat)
            new_row = {'Subject': [subject],
                       'Accuracy': [acc],
                       'Train Accuracy': [trainAcc],
                       'C': [C],
                       'gam': [gamma],
                       'fold': [fold_num]}
            df2 = pd.DataFrame.from_dict(new_row)
            df = pd.concat([df, df2], sort=False)
        fold_num += 1
    # calculate the average accuracy obtained for each C
    # (across all the folds!)
    unique_subjs = df['Subject'].unique()
    df3 = pd.DataFrame()
    for C in C_range:
        # get the sum across all folds
        subj_values = df.loc[(df['C'] == C)]
        avg_acc = np.mean(subj_values['Accuracy'].values)
        mean_train_acc = np.mean(subj_values['Train Accuracy'].values)
        # print(avg_acc)
        new_row2 = {'Subject': [subject],
                    'Accuracy': [avg_acc],
                    'Train Accuracy': [mean_train_acc],
                    'C': [C],
                    'gam': gamma,
                    'Accuracy_Type': ["Validation_Avg"]}
        df4 = pd.DataFrame.from_dict(new_row2)
        df3 = pd.concat([df3, df4], sort=False)
    return df3


def nested_CV_Intra_Subj3(subject, df, scaleTransform):
    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subject)

    # assert train1_labels.dataset == train2_labels.dataset
    # assert train2_labels.dataset == train3_labels.dataset

    # configure the outer loop
    cv_outer = KFold(n_splits=4, shuffle=True, random_state=1)
    # define search space
    global c_POW
    lower_C = c_POW * -1
    upper_C = c_POW
    C_range = np.logspace(lower_C, upper_C, 32)
    gamma_range = np.logspace(lower_C, upper_C, 32)
    # enumerate splits
    fold_num = 1
    for train_ix, test_ix in cv_outer.split(X1_train):
        train_ix = train_ix.astype(dtype=int).tolist()
        test_ix = test_ix.astype(dtype=int).tolist()

        assert train_ix != test_ix

        y_train = torch.utils.data.Subset(train1_labels, train_ix)
        y_validate = torch.utils.data.Subset(train1_labels, test_ix)

        # split data
        X_train1, X_valid1 = X1_train[train_ix, :], X1_train[test_ix, :]
        X_train2, X_valid2 = X2_train[train_ix, :], X2_train[test_ix, :]
        X_train3, X_valid3 = X3_train[train_ix, :], X3_train[test_ix, :]

        X1 = X_train1, X_valid1, X1_test
        X2 = X_train2, X_valid2, X2_test
        X3 = X_train3, X_valid3, X3_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        classifiers = []
        for C in C_range:
            for gamma in gamma_range:
                # Create a model
                clf = SVC(kernel='rbf', class_weight='balanced', C=C, gamma=gamma)
                # Train the model
                clf.fit(X_train, y_train)

                yTrain = [y_train[i] for i, val in enumerate(y_train)]
                # Training accuracy
                trainHat = clf.predict(X_train)
                trainAcc = accuracy_score(yTrain, trainHat)

                # Evaluate the model
                y_valid = [y_validate[i] for i, val in enumerate(y_validate)]
                yhat = clf.predict(X_validate)
                acc = accuracy_score(y_valid, yhat)
                new_row = {'Subject': [subject],
                           'Accuracy': [acc],
                           'Train Accuracy': [trainAcc],
                           'C': [C],
                           'gam': [gamma],
                           'fold': [fold_num]}
                df2 = pd.DataFrame.from_dict(new_row)
                df = pd.concat([df, df2], sort=False)
        fold_num += 1
    # calculate the average accuracy obtained for each pair of parameters C and gamma
    # (across all the folds!)
    unique_subjs = df['Subject'].unique()
    df3 = pd.DataFrame()
    for C in C_range:
        for gamma in gamma_range:
            # get the sum across all folds
            subj_values = df.loc[(df['C'] == C) & (df['gam'] == gamma)]
            avg_acc = np.mean(subj_values['Accuracy'].values)
            mean_train_acc = np.mean(subj_values['Train Accuracy'].values)
            # print(avg_acc)
            new_row2 = {'Subject': [subject],
                        'Accuracy': [avg_acc],
                        'Train Accuracy': [mean_train_acc],
                        'C': [C],
                        'gam': gamma,
                        'Accuracy_Type': ["Validation_Avg"]}
            df4 = pd.DataFrame.from_dict(new_row2)
            df3 = pd.concat([df3, df4], sort=False)
    return df3


def run_indiv_nestedCV(patho, scaleTransform):
    seed = get_seed()
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./{seed}_individual_nestedCV_results'
    global c_POW
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'{pkl_name}_patho_{c_POW}'
    else:
        pkl_name = f'{pkl_name}_{c_POW}'

    if scaleTransform:
        pkl_name = f'{pkl_name}_scaled.pkl'
    else:
        pkl_name = f'{pkl_name}.pkl'

    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    df = pd.DataFrame()
    for entry in index:
        subject = entry['Subject']
        print(f'Running NestedCV for: {subject}')
        df2 = nested_CV_Intra_Subj3(subject, df, scaleTransform)
        df = pd.concat([df, df2], sort=False)
    df.to_pickle(pkl_name)


def run_indiv_fixed_Gamma(patho, scaleTransform):
    seed = get_seed()
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./{seed}_individual_nestedCV_Fixed_Gamma'
    global c_POW
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'{pkl_name}_patho_{c_POW}'
    else:
        pkl_name = f'{pkl_name}_{c_POW}'

    if scaleTransform:
        pkl_name = f'{pkl_name}_scaled.pkl'
    else:
        pkl_name = f'{pkl_name}.pkl'

    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    df = pd.DataFrame()
    for entry in index:
        subject = entry['Subject']
        print(f'Running NestedCV for: {subject}')
        df3 = nested_CV_Intra_Subj3(subject, df, scaleTransform)

        # sort based on accuracy, gam, C
        df3 = df3.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
        # drop duplicates, specify Subject
        df3 = df3.drop_duplicates('Subject', keep='first')
        df3 = df3.sort_values(by=['Subject'])

        subject_Accuracy = df3.Accuracy[0]
        subject_Train_Acc = df3["Train Accuracy"][0]

        subject_C = df3.C[0]
        subj_gam = df3.gam[0]

        df2 = nested_CV_Fixed_Gamma(subject, df, subj_gam)

        df4 = df2.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
        # drop duplicates, specify Subject
        df4 = df4.drop_duplicates('Subject', keep='first')
        df4 = df4.sort_values(by=['Subject'])

        df = pd.concat([df, df2], sort=False)
    df.to_pickle(pkl_name)


# Run a "Vertical" Test - for a fixed (optimal) value of gamma, test the range of C parameters
def range_gamma_fixed_C(subject, df, scaleTransform, gamma_range, C=None):
    print(f'Running increasing gamma for: {subject}, C: {C}')

    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_true_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_true_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_true_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subject)

    # configure the outer loop
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

    outer_idx = 0
    for train_ix, test_ix in cv_outer.split(X1_train):
        train_ix = train_ix.astype(dtype=int).tolist()
        test_ix = test_ix.astype(dtype=int).tolist()

        y_train = torch.utils.data.Subset(train1_labels, train_ix)
        y_validate = torch.utils.data.Subset(train1_labels, test_ix)

        # split data
        X_train1, X_valid1 = X1_train[train_ix, :], X1_train[test_ix, :]
        X_train2, X_valid2 = X2_train[train_ix, :], X2_train[test_ix, :]
        X_train3, X_valid3 = X3_train[train_ix, :], X3_train[test_ix, :]

        X1 = X_train1, X_valid1, X1_true_test
        X2 = X_train2, X_valid2, X2_true_test
        X3 = X_train3, X_valid3, X3_true_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        if C is None:
            C = 1.0

        for gamma in gamma_range:
            def_gam = gamma
            # define the model
            model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
            # fit the model on the inner fold
            model.fit(X_train, y_train)

            # evaluate model on held-out dataset (validation set)
            yhat = model.predict(X_validate)
            # evaluate the model
            y_validate = [y_validate[i] for i, val in enumerate(y_validate)]
            valid_acc = accuracy_score(y_validate, yhat)

            if subject.startswith('C'):
                subj_type = 'Control'
            else:
                subj_type = 'Stutter'

            # report progress
            outer_idx += 1
            # prepare row and append to the dataframe
            new_row = {'Subject': [subject],
                       'Accuracy': [valid_acc],
                       'Subject_Type': [subj_type],
                       'C': [C],
                       'gam': [def_gam],
                       'Accuracy_Type': 'validation'}
            df2 = pd.DataFrame.from_dict(new_row)
            df = pd.concat([df, df2], sort=False)

            # evaluate model on held-out dataset ("true" test set)
            yhat2 = model.predict(X_test)
            # evaluate the model
            y_test = [test1_labels[i] for i, val in enumerate(test1_labels)]
            test_acc = accuracy_score(y_test, yhat2)

            # prepare row and append to the dataframe
            new_row2 = {'Subject': [subject],
                        'Accuracy': [test_acc],
                        'Subject_Type': [subj_type],
                        'C': [C],
                        'gam': [def_gam],
                        'Accuracy_Type': 'test'}
            df3 = pd.DataFrame.from_dict(new_row2)
            df = pd.concat([df, df3], sort=False)

    return df


# Run a "Horizontal" Test - for a fixed (optimal) value of C, test the range of gamma parameters
def range_C_fixed_gamma(subject, df, scaleTransform, C_range, gamma=None):
    print(f'Running increasing C for: {subject}, Gamma: {gamma}')

    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_true_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_true_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_true_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subject)

    # configure the outer loop
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

    lower_C = c_POW * -1
    upper_C = c_POW
    # C_range = np.logspace(1, 12, 16)

    outer_idx = 0
    for train_ix, test_ix in cv_outer.split(X1_train):
        train_ix = train_ix.astype(dtype=int).tolist()
        test_ix = test_ix.astype(dtype=int).tolist()

        y_train = torch.utils.data.Subset(train1_labels, train_ix)
        y_validate = torch.utils.data.Subset(train1_labels, test_ix)

        # split data
        X_train1, X_valid1 = X1_train[train_ix, :], X1_train[test_ix, :]
        X_train2, X_valid2 = X2_train[train_ix, :], X2_train[test_ix, :]
        X_train3, X_valid3 = X3_train[train_ix, :], X3_train[test_ix, :]

        X1 = X_train1, X_valid1, X1_true_test
        X2 = X_train2, X_valid2, X2_true_test
        X3 = X_train3, X_valid3, X3_true_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        X_var = X_train.var()

        if gamma is None:
            # gamma = (1.0 / (180 * X_var)) / 2
            # gamma = 10 ** -7
            gamma = 50
            # def_gam = [(1.0 / (180 * X_var))]
        def_gam = gamma

        for c in C_range:
            # define the model
            # default gamma (scale) == 1 / (n_features * X.var())
            model = SVC(kernel='rbf', C=c, gamma=gamma, class_weight='balanced')
            # fit the model on the inner fold
            model.fit(X_train, y_train)

            # evaluate model on held-out dataset (validation set)
            yhat = model.predict(X_validate)
            # evaluate the model
            y_validate = [y_validate[i] for i, val in enumerate(y_validate)]
            valid_acc = accuracy_score(y_validate, yhat)

            if subject.startswith('C'):
                subj_type = 'Control'
            else:
                subj_type = 'Stutter'

            # report progress
            # print('>valid. acc=%.3f' % valid_acc)
            outer_idx += 1
            # prepare row and append to the dataframe
            new_row = {'Subject': [subject],
                       'Accuracy': [valid_acc],
                       'Subject_Type': [subj_type],
                       'C': [c],
                       'gam': [def_gam],
                       'Accuracy_Type': 'validation'}
            df2 = pd.DataFrame.from_dict(new_row)
            df = pd.concat([df, df2], sort=False)

            # evaluate model on held-out dataset ("true" test set)
            yhat2 = model.predict(X_test)
            # evaluate the model
            y_test = [test1_labels[i] for i, val in enumerate(test1_labels)]
            test_acc = accuracy_score(y_test, yhat2)

            # report progress
            # print('>test acc=%.3f' % test_acc)
            # prepare row and append to the dataframe
            new_row2 = {'Subject': [subject],
                        'Accuracy': [test_acc],
                        'Subject_Type': [subj_type],
                        'C': [c],
                        'gam': [def_gam],
                        'Accuracy_Type': 'test'}
            df3 = pd.DataFrame.from_dict(new_row2)
            df = pd.concat([df, df3], sort=False)

    return df


# gather validation + test accuracy as C increases (fixed default gamma)
def variable_C_fixed_gamma(subject, df, scaleTransform, gamma=None):
    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_true_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_true_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_true_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subject)

    # configure the outer loop
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)

    lower_C = c_POW * -1
    upper_C = c_POW
    C_range = np.logspace(1, 12, 16)

    outer_idx = 0
    for train_ix, test_ix in cv_outer.split(X1_train):
        train_ix = train_ix.astype(dtype=int).tolist()
        test_ix = test_ix.astype(dtype=int).tolist()

        y_train = torch.utils.data.Subset(train1_labels, train_ix)
        y_validate = torch.utils.data.Subset(train1_labels, test_ix)

        # split data
        X_train1, X_valid1 = X1_train[train_ix, :], X1_train[test_ix, :]
        X_train2, X_valid2 = X2_train[train_ix, :], X2_train[test_ix, :]
        X_train3, X_valid3 = X3_train[train_ix, :], X3_train[test_ix, :]

        X1 = X_train1, X_valid1, X1_true_test
        X2 = X_train2, X_valid2, X2_true_test
        X3 = X_train3, X_valid3, X3_true_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        X_var = X_train.var()

        if gamma is None:
            # gamma = (1.0 / (180 * X_var)) / 2
            # gamma = 10 ** -7
            gamma = 50
            # def_gam = [(1.0 / (180 * X_var))]
        def_gam = gamma

        for c in C_range:
            # define the model
            # default gamma (scale) == 1 / (n_features * X.var())
            model = SVC(kernel='rbf', C=c, gamma=gamma, class_weight='balanced')
            # fit the model on the inner fold
            model.fit(X_train, y_train)

            # evaluate model on held-out dataset (validation set)
            yhat = model.predict(X_validate)
            # evaluate the model
            y_validate = [y_validate[i] for i, val in enumerate(y_validate)]
            valid_acc = accuracy_score(y_validate, yhat)

            # report progress
            # print('>valid. acc=%.3f' % valid_acc)
            outer_idx += 1
            # prepare row and append to the dataframe
            new_row = {'Subject': [subject],
                       'Accuracy': [valid_acc],
                       'C': [c],
                       'gam': [def_gam],
                       'Accuracy_Type': 'validation'}
            df2 = pd.DataFrame.from_dict(new_row)
            df = pd.concat([df, df2], sort=False)

            # evaluate model on held-out dataset ("true" test set)
            yhat2 = model.predict(X_test)
            # evaluate the model
            y_test = [test1_labels[i] for i, val in enumerate(test1_labels)]
            test_acc = accuracy_score(y_test, yhat2)

            # report progress
            # print('>test acc=%.3f' % test_acc)
            # prepare row and append to the dataframe
            new_row2 = {'Subject': [subject],
                        'Accuracy': [test_acc],
                        'C': [c],
                        'gam': [def_gam],
                        'Accuracy_Type': 'test'}
            df3 = pd.DataFrame.from_dict(new_row2)
            df = pd.concat([df, df3], sort=False)

    return df


def variable_gamma_fixed_C(subject, df, scaleTransform):
    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_true_test, \
    train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_true_test, \
    train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_true_test, \
    train3_labels, test3_labels = train_test_split_helper(2, subject)

    # configure the outer loop
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # cv_outer = KFold(n_splits=4, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    outer_params = list()

    outer_idx = 0
    for train_ix, test_ix in cv_outer.split(X1_train):
        train_ix = train_ix.astype(dtype=int).tolist()
        test_ix = test_ix.astype(dtype=int).tolist()

        y_train = torch.utils.data.Subset(train1_labels, train_ix)
        y_validate = torch.utils.data.Subset(train1_labels, test_ix)

        # split data
        X_train1, X_valid1 = X1_train[train_ix, :], X1_train[test_ix, :]
        X_train2, X_valid2 = X2_train[train_ix, :], X2_train[test_ix, :]
        X_train3, X_valid3 = X3_train[train_ix, :], X3_train[test_ix, :]

        X1 = X_train1, X_valid1, X1_true_test
        X2 = X_train2, X_valid2, X2_true_test
        X3 = X_train3, X_valid3, X3_true_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        X_var = X_train.var()
        # def_gam = [(1.0 / (180 * X_var))]
        gamma_range = np.logspace(-3, 7, 16)
        # gamma_range = np.append(gamma_range, def_gam)

        for gamm in gamma_range:
            # define the model
            # default gamma (scale) == 1 / (n_features * X.var())
            model = SVC(kernel='rbf', C=1.0, gamma=gamm, class_weight='balanced')
            # fit the model on the inner fold
            model.fit(X_train, y_train)

            # evaluate model on held-out dataset (validation set)
            yhat = model.predict(X_validate)
            # evaluate the model
            y_validate = [y_validate[i] for i, val in enumerate(y_validate)]
            valid_acc = accuracy_score(y_validate, yhat)

            # report progress
            # print('>valid. acc=%.3f' % valid_acc)
            outer_idx += 1
            # prepare row and append to the dataframe
            new_row = {'Subject': [subject],
                       'Accuracy': [valid_acc],
                       'C': [model.C],
                       'gam': [gamm],
                       'Accuracy_Type': 'validation'}
            df2 = pd.DataFrame.from_dict(new_row)
            df = pd.concat([df, df2], sort=False)

            # evaluate model on held-out dataset ("true" test set)
            yhat2 = model.predict(X_test)
            # evaluate the model
            y_test = [test1_labels[i] for i, val in enumerate(test1_labels)]
            test_acc = accuracy_score(y_test, yhat2)

            # report progress
            # print('>test acc=%.3f' % test_acc)
            # prepare row and append to the dataframe
            new_row2 = {'Subject': [subject],
                        'Accuracy': [test_acc],
                        'C': [model.C],
                        'gam': [gamm],
                        'Accuracy_Type': 'test'}
            df3 = pd.DataFrame.from_dict(new_row2)
            df = pd.concat([df, df3], sort=False)

    return df


def run_indiv_increasingC(patho=False, scaleTransform=True):
    seed = get_seed()
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./{seed}_individual_variableC_results'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'{pkl_name}_patho'

    if scaleTransform:
        pkl_name = f'{pkl_name}_scaled.pkl'
    else:
        pkl_name = f'{pkl_name}.pkl'

    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    df = pd.DataFrame()
    for entry in index:
        subject = entry['Subject']
        print(f'Running increasing C accuracy eval for: {subject}')
        df = variable_C_fixed_gamma(subject, df, scaleTransform)
    df.to_pickle(pkl_name)


def run_indiv_increasingGamma(patho, scaleTransform=True):
    seed = get_seed()
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./{seed}_individual_variableGamma_results'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'{pkl_name}_patho'

    if scaleTransform:
        pkl_name = f'{pkl_name}_scaled.pkl'
    else:
        pkl_name = f'{pkl_name}.pkl'

    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    df = pd.DataFrame()
    for entry in index:
        subject = entry['Subject']
        print(f'Running increasing Gamma accuracy eval for: {subject}')
        df = variable_gamma_fixed_C(subject, df, scaleTransform)
    df.to_pickle(pkl_name)


def run_indiv_defaultParams(scaleTransform, patho=False):
    seed = get_seed()
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./{seed}_individual_results'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'{pkl_name}_patho'

    if scaleTransform:
        pkl_name = f'{pkl_name}_scaled.pkl'
    else:
        pkl_name = f'{pkl_name}.pkl'
    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    df = pd.DataFrame()
    for entry in index:
        subject = entry['Subject']
        print(f'Running model training and eval for: {subject}')
        df = default_param_classification(subject, df, scaleTransform)
    df.to_pickle(pkl_name)


# try the RBF SVC w/ default parameters
def default_param_classification(subject, df, scaleTransform):
    return chosen_param_classification(subject, df, scaleTransform, C=1.0, gamma='scale')


def chosen_param_classification(subject, df, scaleTransform, C=1.0, gamma='scale'):
    seed = get_seed()
    (X_train, y_train), (X_test, y_test) = get_data(subject, scaleTransform)

    # define the model
    # default gamma (scale) == 1 / (n_features * X.var())
    model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
    # fit the model
    model.fit(X_train, y_train)

    X_var = X_train.var()
    if gamma == 'scale':
        def_gam = [(1.0 / (180 * X_var))]
    else:
        def_gam = gamma

    # evaluate model on held-out dataset ("true" test set)
    yhat = model.predict(X_test)
    # evaluate the model
    test_acc = accuracy_score(y_test, yhat)

    # evaluate model on training dataset (sanity check - training accuracy)
    y_train_hat = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_hat)

    debug_print_stmts = False
    if debug_print_stmts:
        print(f"Train Accuracy: {train_acc}")
        print(f"Gamma: {def_gam}")

    report = classification_report(y_test, yhat)

    with open(f'./results/class_reports/{subject}_C_{C}_G_{gamma}_seed_{seed}.txt', 'w') as handle:
        handle.write(report)

    # report progress
    # print('>test acc=%.3f' % test_acc)
    subj_type = 'Control'
    if subject.startswith('S'):
        subj_type = 'Stutter'
    # prepare row and append to the dataframe
    new_row2 = {'Subject': [subject],
                'Accuracy': [test_acc],
                'Subject_Type': [subj_type],
                'C': [C],
                'gam': [def_gam],
                'Accuracy_Type': 'test'}
    df3 = pd.DataFrame.from_dict(new_row2)
    df = pd.concat([df, df3], sort=False)

    return df


# Select 75% of the training data to test the classifier performance with less training data!
def small_data_classification(subject, df, scaleTransform, C=1.0, gamma='scale'):
    (X_train, y_train), (X_test, y_test) = get_data(subject, scaleTransform)

    train_size = int(len(X_train) * 0.75)
    X_train = X_train[0:train_size]
    y_train = y_train[0:train_size]

    # define the model
    # default gamma (scale) == 1 / (n_features * X.var())
    model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
    # fit the model
    model.fit(X_train, y_train)

    X_var = X_train.var()
    if gamma == 'scale':
        def_gam = [(1.0 / (180 * X_var))]
    else:
        def_gam = gamma

    # evaluate model on held-out dataset ("true" test set)
    yhat = model.predict(X_test)
    # evaluate the model
    test_acc = accuracy_score(y_test, yhat)

    # evaluate model on training dataset (sanity check - training accuracy)
    y_train_hat = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_hat)
    debug_print_stmts = True
    if debug_print_stmts:
        print(f"Train Accuracy: {train_acc}")
        print(f"Gamma: {def_gam}")
    report = classification_report(y_test, yhat)

    # report progress
    # print('>test acc=%.3f' % test_acc)
    subj_type = 'Control'
    if subject.startswith('S'):
        subj_type = 'Stutter'
    # prepare row and append to the dataframe
    new_row2 = {'Subject': [subject],
                'Accuracy': [test_acc],
                'Subject_Type': [subj_type],
                'C': [C],
                'gam': [def_gam],
                'Accuracy_Type': 'small_test'}
    df3 = pd.DataFrame.from_dict(new_row2)
    df = pd.concat([df, df3], sort=False)

    return df


def run_classification_all_groups_default_params(scaleTransform):
    run_indiv_defaultParams(scaleTransform, patho=False)
    run_indiv_defaultParams(scaleTransform, patho=True)
    combine_df_results_defaultParams(scaleTransform)


def run_single_param_increasing(scaleTransform):
    run_indiv_increasingC(False, scaleTransform)
    run_indiv_increasingC(True, scaleTransform)
    combine_df_results_increasingC(scaleTransform)

    run_indiv_increasingGamma(False, scaleTransform)
    run_indiv_increasingGamma(True, scaleTransform)
    combine_df_results_increasingGamma(scaleTransform)


def run_nestedCV_all(scaleTransform):
    run_indiv_nestedCV(False, scaleTransform)
    run_indiv_nestedCV(True, scaleTransform)
    combine_df_nestedCV(scaleTransform)


def hyp_tuning(scaleTransform):
    # run_classification_all_groups_default_params(scaleTransform)
    # run_single_param_increasing(scaleTransform)
    c_pow_range = [1, 3, 5, 7]
    for c in c_pow_range:
        global c_POW
        c_POW = c
        run_nestedCV_all(scaleTransform)


# Run increasing C trials using the optimal gamma parameter found through tuning
def evaluate_gamma(scaleTransform):
    seed = get_seed()
    pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_combined.pkl'
    out_pkl_name = f'./{seed}_hyperparam_testing_{c_POW}.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_scaled_combined.pkl'
        out_pkl_name = f'./{seed}_hyperparam_testing_{c_POW}_scaled.pkl'
    df = pd.read_pickle(pkl_name)
    # sort based on accuracy, gam, C
    df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
    # drop duplicates, specify Subject
    df = df.drop_duplicates('Subject', keep='first')
    unique_subjs = df['Subject'].unique()
    # max accuracy achieved per-subject,
    # corresp. C and gamma params
    # df2 = pd.DataFrame()
    if True:  # not os.path.exists(f'./{seed}_optimal_gam_Increasing_C_scaled_{c_POW}.pkl'):
        df3 = pd.DataFrame()
        C_range = np.logspace((-1 * c_POW), c_POW, 32)
        for subj in unique_subjs:
            subj_df_loc = df.loc[df['Subject'] == subj]
            valid_acc = subj_df_loc.Accuracy[0]
            # subj_gam = subj_df_loc.gam[0]
            subj_gam = 0.005
            df3 = range_C_fixed_gamma(subj, df3, scaleTransform, C_range, gamma=subj_gam)
        df3.to_pickle(f'./{seed}_optimal_gam_Increasing_C_scaled_{c_POW}.pkl')
    # plot_opt_gam_increasing_C(f'./{seed}_optimal_gam_Increasing_C_scaled_{c_POW}.pkl')


# Run increasing gamma trials using the optimal gamma parameter found through tuning
def evaluate_C(scaleTransform):
    seed = get_seed()
    pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_combined.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_scaled_combined.pkl'

    df = pd.read_pickle(pkl_name)
    # sort based on accuracy, gam, C
    df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
    # drop duplicates, specify Subject
    df = df.drop_duplicates('Subject', keep='first')
    unique_subjs = df['Subject'].unique()
    if True:  # not os.path.exists(f'./{seed}_optimal_c_Increasing_Gamma_scaled_{c_POW}.pkl'):
        df3 = pd.DataFrame()
        gamma_range = np.logspace((-1 * c_POW), c_POW, 32)
        for subj in unique_subjs:
            subj_df_loc = df.loc[df['Subject'] == subj]
            valid_acc = subj_df_loc.Accuracy[0]
            # subject_C = subj_df_loc.C[0]
            subject_C = 1.0
            df3 = range_gamma_fixed_C(subj, df3, scaleTransform, gamma_range, C=subject_C)
        df3.to_pickle(f'./{seed}_optimal_c_Increasing_Gamma_scaled_{c_POW}.pkl')


def test_evaluation(scaleTransform):
    seed = get_seed()
    pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_combined.pkl'
    out_pkl_name = f'./{seed}_hyperparam_testing_{c_POW}.pkl'
    if scaleTransform:
        pkl_name = f'./{seed}_individual_nestedCV_results_{c_POW}_scaled_combined.pkl'
        out_pkl_name = f'./{seed}_hyperparam_testing_{c_POW}_scaled.pkl'
    df = pd.read_pickle(pkl_name)
    # sort based on accuracy, gam, C
    df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
    # drop duplicates, specify Subject
    df = df.drop_duplicates('Subject', keep='first')
    unique_subjs = df['Subject'].unique()
    # max accuracy achieved per-subject,
    # corresp. C and gamma params
    df2 = pd.DataFrame()
    # df3 = pd.DataFrame()
    # C_range = np.logspace((-1 * c_POW), c_POW, 32)
    for subj in unique_subjs:
        subj_df_loc = df.loc[df['Subject'] == subj]
        valid_acc = subj_df_loc.Accuracy[0]
        subject_C = subj_df_loc.C[0]
        subj_gam = subj_df_loc.gam[0]
        # df3 = range_C_fixed_gamma(subj, df3, scaleTransform, C_range, gamma=subj_gam)
        df2 = chosen_param_classification(subj, df2, scaleTransform, C=subject_C, gamma=subj_gam)
    df2 = df2.sort_values(by=['Subject'])
    df2.to_pickle(out_pkl_name)
    # df3.to_pickle(f'./{seed}_optimal_gam_Increasing_C_scaled.pkl')
    # plot_opt_gam_increasing_C()


def logistic_regression(subj, df):
    (X_train, y_train), (X_test, y_test) = get_data(subj, True)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression ({subj}) Test Accuracy: {acc}')
    subj_type = 'Control'
    if subj.startswith('S'):
        subj_type = 'Stutter'
    # prepare row and append to the dataframe
    new_row2 = {'Subject': [subj],
                'Accuracy': [acc],
                'Subject_Type': [subj_type],
                'Accuracy_Type': 'test'}
    df3 = pd.DataFrame.from_dict(new_row2)
    df = pd.concat([df, df3], sort=False)
    return df


def run_indiv_log_regression(patho=False):
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./individual_log_regression_results.pkl'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'./individual_log_regression_results_patho.pkl'
    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    df = pd.DataFrame()
    for entry in index:
        subject = entry['Subject']
        df = logistic_regression(subject, df)
    df.to_pickle(pkl_name)
