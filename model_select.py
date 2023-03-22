import json
import numpy as np

from eeg_dataset import EEGSubjectDataset
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold


# Implement nested cross-validation within a single subject
# Inspired by: https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/
def nested_CV_Intra_Subj(subject):
    # Load the dataset
    eeg_train = EEGSubjectDataset(subject, train=True)
    # Create data loaders
    train_loader = DataLoader(eeg_train, batch_size=64, shuffle=True)
    X = train_loader.dataset.data
    y = train_loader.dataset.targets
    # configure the outer loop
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    # cv_outer = KFold(n_splits=4, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    X_var = X.var()
    def_C = [1.0]
    def_gam = [(1.0 / (180 * X_var))]
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the inner loop
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
        # define the model
        model = SVC(kernel='rbf', class_weight='balanced')
        # define search space (note: includes default values!)
        C_range = np.logspace(-2, 2, 16)
        C_range = np.append(C_range, def_C)
        gamma_range = np.logspace(-3, 4, 16)
        gamma_range = np.append(gamma_range, def_gam)
        # C_range = np.logspace(-2, 10, 16)
        # gamma_range = np.logspace(-9, 3, 16)
        param_grid = dict(gamma=gamma_range, C=C_range)
        # define search
        search = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model
        best_model = result.best_estimator_
        # evaluate model on held-out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    # summarize the estimated model performance
    print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))


def run_indiv_nestedCV(patho=False):
    base_dir = 'D:\\Research\\EEG-DIVA\\Feature_Selection_PCA'
    index_file = f'{base_dir}\\C_index.json'
    if patho:
        index_file = f'{base_dir}\\S_index.json'
    f = open(index_file)
    index = json.load(f)['index']
    f.close()
    for entry in index:
        subject = entry['Subject']
        print(f'Running NestedCV for: {subject}')
        nested_CV_Intra_Subj(subject)


if __name__ == '__main__':
    run_indiv_nestedCV(patho=False)
