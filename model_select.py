import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import seaborn as sns
import pandas as pd

# from eeg_dataset import EEGSubjectDataset
from EEGSubjectDatasetLarge import EEGSubjectDatasetLarge
from torch.utils.data import DataLoader
import torch
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

sns.set_theme(style="darkgrid")

c_POW = 3


def train_test_split_helper(feat_type, subject):
    patho = False
    if subject.startswith('S'):
        patho = True

    full_dataset = EEGSubjectDatasetLarge(feat_type, subject, patho=patho)
    # perform train/test split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset,
                                      [train_size, test_size],
                                      generator=torch.Generator().manual_seed(42))
    train_labels, test_labels = \
        torch.utils.data.random_split(full_dataset.labels,
                                      [train_size, test_size],
                                      generator=torch.Generator().manual_seed(42))

    X_train = np.array([x for x, y in train_dataset])
    X_test = np.array([x for x, y in test_dataset])

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

    # normalize data
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
    else:
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


# Implement nested cross-validation w/ FEATURE SELECTION within a single subject
# Inspired by: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
def nested_CV_Intra_Subj2(subject, df, scaleTransform):
    # for each subject, there are three LARGE feature sets f0-f2
    X1_train, X1_test, \
        train1_labels, test1_labels = train_test_split_helper(0, subject)

    X2_train, X2_test, \
        train2_labels, test2_labels = train_test_split_helper(1, subject)

    X3_train, X3_test, \
        train3_labels, test3_labels = train_test_split_helper(2, subject)

    assert train1_labels.dataset == train2_labels.dataset
    assert train2_labels.dataset == train3_labels.dataset

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

        X1 = X_train1, X_valid1, X1_test
        X2 = X_train2, X_valid2, X2_test
        X3 = X_train3, X_valid3, X3_test
        X_train, X_validate, X_test = normalize_combine_data(X1, X2, X3, scaleTransform)

        # define search space
        lower_C = c_POW * -1
        upper_C = c_POW
        C_range = np.logspace(lower_C, upper_C, 32)
        gamma_range = np.logspace(lower_C, upper_C, 32)

        # configure the inner loop
        param_grid = dict(gamma=gamma_range, C=C_range)
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = SVC(kernel='rbf', class_weight='balanced')
        # define search
        search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)

        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model
        best_model = result.best_estimator_
        # evaluate model on held-out dataset
        yhat = best_model.predict(X_validate)
        # evaluate the model
        y_valid = [y_validate[i] for i, val in enumerate(y_validate)]
        acc = accuracy_score(y_valid, yhat)
        # store the result
        outer_results.append(acc)
        outer_params.append(result.best_params_)
        # report progress
        # print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
        outer_idx += 1
        # prepare row and append to the dataframe
        new_row = {'Subject': [subject],
                   'Accuracy': [acc],
                   'C': [result.best_params_['C']],
                   'gam': [result.best_params_['gamma']]}
        df2 = pd.DataFrame.from_dict(new_row)
        df = pd.concat([df, df2], sort=False)
    # summarize the estimated model performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))
    return df


def run_indiv_nestedCV(patho, scaleTransform):
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./individual_nestedCV_results'

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
        df = nested_CV_Intra_Subj2(subject, df, scaleTransform)
    df.to_pickle(pkl_name)


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


def extract_max_plot_results(patho=True):
    # load pickle
    pkl_name = f'./individual_nestedCV_results_{c_POW}.pkl'
    if patho:
        pkl_name = f'./individual_nestedCV_results_patho_{c_POW}.pkl'
    df = pd.read_pickle(pkl_name)
    # sort based on accuracy
    df = df.sort_index(1)
    # drop duplicates, specify Subject
    # keep first (last?)
    df = df.drop_duplicates('Subject', keep='last')
    # now, we have max accuracy achieved per-subject
    sns.relplot(data=df, x="Subject", y="Accuracy", kind="line")
    plt.show()


def combine_df_nestedCV(scaleTransform):
    pkl1 = f'./individual_nestedCV_results_{c_POW}'
    pkl2 = f'./individual_nestedCV_results_patho_{c_POW}'

    if scaleTransform:
        pkl1 = f'{pkl1}_scaled.pkl'
        pkl2 = f'{pkl2}_scaled.pkl'

    combine_df_results(pkl1, pkl2)


def combine_df_log_regression():
    combine_df_results(f'./individual_log_regression_results.pkl', f'./individual_log_regression_results_patho.pkl')


def test_mean_significance(pkl_name, attr_name="Accuracy"):
    df = pd.read_pickle(pkl_name)
    # get accuracies of people who stutter
    cond1 = (df['Subject_Type'] == "Stutter")
    pws_acc = df[cond1][attr_name].values
    # get accuracies of people who do not stutter
    cond2 = (df['Subject_Type'] == "Control")
    control_acc = df[cond2][attr_name].values
    mean_AWS = np.mean(pws_acc)
    mean_AWDS = np.mean(control_acc)
    std_dev_AWS = np.std(pws_acc)
    std_dev_AWDS = np.std(control_acc)
    print("Testing if mean difference is statistically significant")
    print("Attribute: %s" % attr_name)

    print(f"AWS Mean (Std): {mean_AWS} ({std_dev_AWS})")
    print(f"AWDS Mean (Std): {mean_AWDS} ({std_dev_AWDS})")

    t_stat, p_val = stats.ttest_ind(control_acc, pws_acc, equal_var=False)
    print('t_stat (p_val): %.3f (%.3f)' % (t_stat, p_val))
    if p_val < 0.05:
        print("SIGNIFICANT RESULT")
    else:
        print("RESULT NOT SIGNIFICANT")


def bar_chart(pkl_name, title=None, logit=False):
    df = pd.read_pickle(pkl_name)

    if title is None:
        title = ""

    def logit_transform(x):
        return scipy.special.logit(x)

    if logit:
        df['Accuracy'] = df['Accuracy'].apply(logit_transform)

    g = sns.catplot(data=df, kind="bar", alpha=.6,
                    x="Subject", y="Accuracy", hue="Subject_Type", height=6, aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    g.despine(left=True)
    plt.show()


def extract_avg_plot_results(patho=False):
    # load pickle
    pkl_name = f'./individual_nestedCV_results_{c_POW}.pkl'
    if patho:
        pkl_name = f'./individual_nestedCV_results_patho_{c_POW}.pkl'
    df = pd.read_pickle(pkl_name)
    # get the unique subjects
    unique_subjs = df['Subject'].unique()
    # drop duplicates, specify Subject
    subj_avgs = []
    for subj in unique_subjs:
        subj_values = df.loc[df['Subject'] == subj]
        avg = np.mean(subj_values['Accuracy'])
        subj_avgs.append(avg)
    # now, we have avg accuracy per-subject
    data = {'Subject': unique_subjs,
            'Average': subj_avgs}
    df2 = pd.DataFrame(data)
    # sns.relplot(data=df2, x="Subject", y="Average", kind="line")
    sns.relplot(data=df, x="Subject", y="Accuracy", kind="line", height=6, aspect=2.5)
    plt.show()


def group_avg_bar_chart_nestedCV(pkl_name, logit=False):
    if logit:
        group_avg_bar_chart(pkl_name, title=f"Logit Group Average Comparison Max-C: 10^{c_POW}", logit=True)
    else:
        group_avg_bar_chart(pkl_name, title=f"Group Average Comparison Max-C: 10^{c_POW}", logit=False)


def group_avg_bar_chart(pkl_name, title=None, logit=False):
    df = pd.read_pickle(pkl_name)

    def logit_transform(x):
        return scipy.special.logit(x)

    if title is None:
        title = ""

    if logit:
        # get accuracies of people who stutter
        cond1 = (df['Subject_Type'] == "Stutter")
        pws_acc = df[cond1].Accuracy.values
        # get accuracies of people who do not stutter
        cond2 = (df['Subject_Type'] == "Control")
        control_acc = df[cond2].Accuracy.values
        pws_avg = np.mean(pws_acc)
        control_avg = np.mean(control_acc)

        pws_avg = logit_transform(pws_avg)
        control_avg = logit_transform(control_avg)

        data = {'Subject_Type': ['Control', 'Stutter'],
                'Mean_Accuracy_logit': [control_avg, pws_avg]}
        df2 = pd.DataFrame(data)
        g = sns.catplot(data=df2, kind="bar", palette="dark", alpha=.6,
                        x="Subject_Type", y="Mean_Accuracy_logit", height=6, aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        g.ax.set_title(title)
        g.despine(left=True)
        plt.ylim(2.0, 3.0)
        plt.show()
    else:
        g = sns.catplot(data=df, kind="bar", palette="dark", alpha=.6,
                        x="Subject_Type", y="Accuracy", height=6, aspect=2.5)
        g.fig.subplots_adjust(top=.95)
        g.ax.set_title(title)
        g.despine(left=True)
        # plt.ylim(0.8, 1.0)
        plt.show()


def normalize_combine_data(X1, X2, X3, scaleTransform):
    X_train1, X_valid1, X_test1 = X1
    X_train2, X_valid2, X_test2 = X2
    X_train3, X_valid3, X_test3 = X3
    # normalize data
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
    else:
        X_train1 = preprocessing.normalize(X_train1, axis=0)
        X_train2 = preprocessing.normalize(X_train2, axis=0)
        X_train3 = preprocessing.normalize(X_train3, axis=0)

        X_valid1 = preprocessing.normalize(X_valid1, axis=0)
        X_valid2 = preprocessing.normalize(X_valid2, axis=0)
        X_valid3 = preprocessing.normalize(X_valid3, axis=0)

        X1_true_test = preprocessing.normalize(X_test1, axis=0)
        X2_true_test = preprocessing.normalize(X_test2, axis=0)
        X3_true_test = preprocessing.normalize(X_test3, axis=0)

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


# gather validation + test accuracy as C increases (fixed default gamma)
def variable_C_fixed_gamma(subject, df, scaleTransform):
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
    C_range = np.logspace(-3, 7, 32)

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
        def_gam = [(1.0 / (180 * X_var))]

        for c in C_range:
            # define the model
            # default gamma (scale) == 1 / (n_features * X.var())
            model = SVC(kernel='rbf', C=c, gamma='scale', class_weight='balanced')
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
        gamma_range = np.logspace(-3, 7, 32)
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
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./individual_variableC_results'
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
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./individual_variableGamma_results'
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
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    pkl_name = f'./individual_results'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
        pkl_name = f'./individual_results_patho'

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


def combine_df_results(pkl1_name, pkl2_name):
    df1 = pd.read_pickle(pkl1_name)
    non_stut_labels = ["Control" for row in df1['Subject']]
    df1["Subject_Type"] = non_stut_labels
    df2 = pd.read_pickle(pkl2_name)
    stut_labels = ["Stutter" for row in df2['Subject']]
    df2["Subject_Type"] = stut_labels
    df = pd.concat([df1, df2], sort=False)
    name_no_ext = os.path.splitext(pkl1_name)[0]
    final_pkl_name = name_no_ext + '_combined.pkl'
    df.to_pickle(final_pkl_name)


def combine_df_results_defaultParams(scaleTransform):
    if scaleTransform:
        combine_df_results('./individual_results_scaled.pkl', './individual_results_patho_scaled.pkl')
    else:
        combine_df_results('./individual_results.pkl', './individual_results_patho.pkl')


def combine_df_results_increasingC(scaleTransform):
    pkl1 = './individual_variableC_results'
    pkl2 = './individual_variableC_results_patho'
    if scaleTransform:
        pkl1 = f'{pkl1}_scaled'
        pkl2 = f'{pkl2}_scaled'
    pkl1 = f'{pkl1}.pkl'
    pkl2 = f'{pkl2}.pkl'
    combine_df_results(pkl1, pkl2)


def combine_df_results_increasingGamma(scaleTransform):
    pkl1 = './individual_variableGamma_results'
    pkl2 = './individual_variableGamma_results_patho'
    if scaleTransform:
        pkl1 = f'{pkl1}_scaled'
        pkl2 = f'{pkl2}_scaled'
    pkl1 = f'{pkl1}.pkl'
    pkl2 = f'{pkl2}.pkl'
    combine_df_results(pkl1, pkl2)


# calculate average params for both groups
def get_avg_params(pkl_name):
    df = pd.read_pickle(pkl_name)
    # get only people who stutter
    cond1 = (df['Subject_Type'] == "Stutter")
    pws_C = df[cond1].C.values
    pws_Gamma = df[cond1].gam.values
    avg_pws_C = np.mean(pws_C)
    avg_pws_gam = np.mean(pws_Gamma)
    # get only people who do not stutter
    cond2 = (df['Subject_Type'] == "Control")
    control_C = df[cond2].C.values
    pws_Gamma = df[cond2].gam.values
    avg_control_C = np.mean(control_C)
    avg_control_gamma = np.mean(pws_Gamma)

    print(f'Control params: C: {avg_control_C}, gamma: {avg_control_gamma}')
    print(f'PWS params: C: {avg_pws_C}, gamma: {avg_pws_gam}')


# plot 2D distribution of selected params across both groups
def plot_params_2D(pkl_name):
    df = pd.read_pickle(pkl_name)
    min_C = df['C'].min()
    max_C = df['C'].max()
    min_gam = df['gam'].min()
    max_gam = df['gam'].max()
    # get the unique subjects
    unique_subjs = df['Subject'].unique()
    # drop duplicates, specify Subject
    subj_avgs = []
    subj_Cs = []
    subj_gams = []
    for subj in unique_subjs:
        subj_values = df.loc[df['Subject'] == subj]
        avg = np.mean(subj_values['Accuracy'])
        avg_C = np.mean(subj_values['C'])
        avg_gam = np.mean(subj_values['gam'])
        subj_avgs.append(avg)
        subj_Cs.append(avg_C)
        subj_gams.append(avg_gam)
    # now, we have avg accuracy per-subject
    data = {'Subject': unique_subjs,
            'Accuracy': subj_avgs,
            'C': subj_Cs,
            'gamma': subj_gams}
    df2 = pd.DataFrame(data)
    df2 = df2.pivot("Subject", "C", "gamma")
    # g1 = sns.heatmap(data=df2, annot=True, fmt=".1f")
    g = sns.scatterplot(data=df, x="C", y="gam", hue="Subject_Type", style="Subject_Type", palette="dark")
    title = f"Per-Subject Params Max-C: 10^{c_POW}"
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    plt.show()


# plot C and gamma average on same plot
def plot_param_avgs(pkl_name):
    df = pd.read_pickle(pkl_name)
    min_C = df['C'].min()
    max_C = df['C'].max()
    min_gam = df['gam'].min()
    max_gam = df['gam'].max()

    # print(f"C: [{min_C}, {max_C}]")
    # print(f"gamma: [{min_gam}, {max_gam}]")
    # g = sns.pairplot(data=df, hue="Subject_Type", palette="dark")
    g = sns.displot(data=df, x="C", y="gam", hue="Subject_Type", kind="kde", palette="dark")
    plt.show()


# plot validation accuracy as a function of C for both control and stutter subjects
def plot_valid_accuracy_vs_C(scaleTransform):
    pkl_name = f'./individual_variableC_results_combined.pkl'
    if scaleTransform:
        pkl_name = './individual_variableC_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'validation']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="C", y="Accuracy", hue="Subject_Type", palette="dark", height=6,
                    aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title("Validation Accuracy")
    plt.show()


# plot test accuracy as a function of C for both control and stutter subjects
def plot_test_accuracy_vs_C(scaleTransform):
    pkl_name = f'./individual_variableC_results_combined.pkl'
    if scaleTransform:
        pkl_name = './individual_variableC_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'test']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="C", y="Accuracy", hue="Subject_Type", palette="dark", height=6,
                    aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title("Test Accuracy")
    plt.show()


# plot validation accuracy as a function of gamma for both control and stutter subjects
def plot_valid_accuracy_vs_gamma(scaleTransform):
    pkl_name = f'./individual_variableGamma_results_combined.pkl'
    if scaleTransform:
        pkl_name = './individual_variableGamma_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'validation']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="gam", y="Accuracy", hue="Subject_Type", palette="dark", height=6,
                    aspect=2.5)
    title = "Validation Accuracy"
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title(title)
    plt.show()


# plot test accuracy as a function of gamma for both control and stutter subjects
def plot_test_accuracy_vs_gamma(scaleTransform):
    pkl_name = f'./individual_variableGamma_results_combined.pkl'
    if scaleTransform:
        pkl_name = './individual_variableGamma_results_scaled_combined.pkl'
    df = pd.read_pickle(pkl_name)
    # select only the validation accuracy
    df = df[df['Accuracy_Type'] == 'test']
    # plot the accuracy curve
    g = sns.relplot(data=df, kind="line", x="gam", y="Accuracy", hue="Subject_Type", palette="dark", height=6,
                    aspect=2.5)
    g.fig.subplots_adjust(top=.95)
    g.ax.set_title("Test Accuracy")
    plt.show()


# try the RBF SVC w/ default parameters
def default_param_classification(subject, df, scaleTransform):
    return chosen_param_classification(subject, df, scaleTransform, C=1.0, gamma='scale')


def chosen_param_classification(subject, df, scaleTransform, C=1.0, gamma='scale'):
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

    report = classification_report(y_test, yhat)

    with open(f'./results/class_reports/{subject}_C_{C}_G_{gamma}.txt', 'w') as handle:
        handle.write(report)

    # report progress
    print('>test acc=%.3f' % test_acc)
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


def visualize_decision_boundary(subject, scaleTransform):
    (X_train, y_train), (X_test, y_test) = get_data(subject, scaleTransform)

    # define the dataset for decision function visualization: keep only the first two features in X
    X_2d = X_train[:, 3:5]
    y_2d = y_train
    C_2d_range = [1e-2, 1, 1e7]
    gamma_2d_range = [1e-1, 1, 1e7]

    # lower_C = c_POW * -1
    # upper_C = c_POW
    # C_2d_range = np.logspace(lower_C, upper_C, 3)
    # gamma_2d_range = np.logspace(lower_C, upper_C, 3)

    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf))
    plt.figure(figsize=(8, 6))
    fig_span = np.abs(np.max(X_2d) - np.min(X_2d))
    quart_span = fig_span / 8
    xx, yy = np.meshgrid(np.linspace(np.min(X_2d) - quart_span, np.max(X_2d), 200) + quart_span,
                         np.linspace(np.min(X_2d) - quart_span, np.max(X_2d) + quart_span, 200))
    for k, (C, gamma, clf) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)), size="medium")

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r, edgecolors="k")
        plt.xticks(())
        plt.yticks(())
        plt.axis("tight")
    plt.show()


def run_classification_all_groups_default_params(scaleTransform):
    run_indiv_defaultParams(scaleTransform, patho=False)
    run_indiv_defaultParams(scaleTransform, patho=True)
    combine_df_results_defaultParams(scaleTransform)
    plot_indiv_and_group_test_acc(scaleTransform)


def plot_single_param_increasing(scaleTransform):
    # Plot accuracy as a function of C parameter
    run_indiv_increasingC(False, scaleTransform)
    run_indiv_increasingC(True, scaleTransform)
    combine_df_results_increasingC(scaleTransform)
    plot_valid_accuracy_vs_C(scaleTransform)
    plot_test_accuracy_vs_C(scaleTransform)
    # Plot accuracy as a function of gamma parameter
    run_indiv_increasingGamma(False, scaleTransform)
    run_indiv_increasingGamma(True, scaleTransform)
    combine_df_results_increasingGamma(scaleTransform)
    plot_valid_accuracy_vs_gamma(scaleTransform)
    plot_test_accuracy_vs_gamma(scaleTransform)


def plot_indiv_and_group_test_acc(scaleTransform=False):
    # plot average test accuracy for all with default params
    indiv_results_pkl = './individual_results_combined.pkl'
    if scaleTransform:
        indiv_results_pkl = './individual_results_scaled_combined.pkl'
    bar_chart(indiv_results_pkl, title="Individual Test Accuracy Default Params", logit=False)
    group_avg_bar_chart(indiv_results_pkl, title="Group Mean Test Accuracy Default Params", logit=False)
    test_mean_significance(indiv_results_pkl)


def run_nestedCV_all(scaleTransform):
    run_indiv_nestedCV(False, scaleTransform)
    run_indiv_nestedCV(True, scaleTransform)
    combine_df_nestedCV(scaleTransform)


def plot_current_cPOW(scaleTransform):
    pkl_name = f'./individual_nestedCV_results_{c_POW}_combined.pkl'
    if scaleTransform:
        pkl_name = f'./individual_nestedCV_results_{c_POW}_scaled_combined.pkl'
    bar_chart(pkl_name,
              title=f"Average Validation Accuracy Comparison Max-C: 10^{c_POW}", logit=False)
    group_avg_bar_chart(pkl_name, title=f"Group Mean Validation Accuracy Comparison, Max-C: 10^{c_POW}", logit=False)
    # group_avg_bar_chart(pkl_name, logit=True)
    plot_params_2D(pkl_name)
    # print("Average Parameters")
    # get_avg_params(pkl_name)
    # print(f"Significance testing for c_POW: {c_POW}")
    # test_mean_significance(pkl_name)
    # test_mean_significance(pkl_name, 'C')
    # test_mean_significance(pkl_name, 'gam')
    # plot_param_avgs(pkl_name)


def hyp_tuning_and_validation(scaleTransform):
    run_classification_all_groups_default_params(scaleTransform)
    plot_single_param_increasing(scaleTransform)
    c_pow_range = [3]
    for c in c_pow_range:
        c_POW = c
        run_nestedCV_all(scaleTransform)
        test_evaluation(scaleTransform)
        pkl_name = f'./hyperparam_testing_{c_POW}.pkl'
        if scaleTransform:
            pkl_name = f'./hyperparam_testing_{c_POW}_scaled.pkl'
        bar_chart(pkl_name, title=f"Test Evaluation with Opt. Params (Range: 10^{c_POW})")
        group_avg_bar_chart(pkl_name, title=f"Grouped Test Evaluation with Opt. Params (Range: 10^{c_POW})")
        test_mean_significance(pkl_name)
        test_mean_significance(pkl_name, 'C')
        test_mean_significance(pkl_name, 'gam')
        plot_current_cPOW(scaleTransform)


def test_evaluation(scaleTransform):
    pkl_name = f'./individual_nestedCV_results_{c_POW}_combined.pkl'
    out_pkl_name = f'./hyperparam_testing_{c_POW}.pkl'
    if scaleTransform:
        pkl_name = f'./individual_nestedCV_results_{c_POW}_scaled_combined.pkl'
        out_pkl_name = f'./hyperparam_testing_{c_POW}_scaled.pkl'
    df = pd.read_pickle(pkl_name)
    # sort based on accuracy, gam, C
    df = df.sort_values(by=['Accuracy', 'gam', 'C'], ascending=[False, True, True])
    # drop duplicates, specify Subject
    df = df.drop_duplicates('Subject', keep='first')
    unique_subjs = df['Subject'].unique()
    # max accuracy achieved per-subject,
    # corresp. C and gamma params
    df2 = pd.DataFrame()
    for subj in unique_subjs:
        subj_df_loc = df.loc[df['Subject'] == subj]
        valid_acc = subj_df_loc.Accuracy[0]
        subject_C = subj_df_loc.C[0]
        subj_gam = subj_df_loc.gam[0]
        df2 = chosen_param_classification(subj, df2, scaleTransform, C=subject_C, gamma=subj_gam)
    df2 = df2.sort_values(by=['Subject'])
    df2.to_pickle(out_pkl_name)


def logistic_regression(subj, df):
    (X_train, y_train), (X_test, y_test) = get_data(subj)
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


def decision_boundary_tests():
    # run_classification_all_groups_default_params(scaleTransform=False)
    # pkl_name = f'./individual_results_combined.pkl'
    # bar_chart(pkl_name, title='SVC Individual Accuracy')
    # group_avg_bar_chart(pkl_name, title='SVC Group Mean Accuracy')
    # run_classification_all_groups_default_params(scaleTransform=True)
    # pkl_name = f'./individual_results_scaled_combined.pkl'
    # bar_chart(pkl_name, title='SVC Individual Accuracy - Alternate Scaling')
    # group_avg_bar_chart(pkl_name, title='SVC Group Mean Accuracy - Alternate Scaling')
    subject = 'CF60'
    visualize_decision_boundary(subject, scaleTransform=False)
    # scaleTransform = True
    # visualize_decision_boundary(subject, scaleTransform)


def log_regression_tests():
    scaleTransform = False
    run_indiv_log_regression(patho=False)
    run_indiv_log_regression(patho=True)
    combine_df_log_regression()
    pkl_name = './individual_log_regression_results_combined.pkl'
    bar_chart(pkl_name, title="Logistic Regression Accuracy")
    group_avg_bar_chart(pkl_name, title="Logistic Regression Group Mean Accuracy")


# Note: scaleTransform=True!!
if __name__ == '__main__':
    scaleTransform = False
    hyp_tuning_and_validation(scaleTransform)
    # decision_boundary_tests()
    # log_regression_tests()
    # param_tuning_tests()
