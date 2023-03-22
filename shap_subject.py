import pickle
import os
import pandas as pd
import numpy as np
import weightedSHAP
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from torch.utils.data import DataLoader

import torch


if __name__ == '__main__':
    df = pd.read_pickle('./eeg_dataset.pkl')
    #print(df)
    debug = False
    train_all = df[df["train_test"] == "train"]
    test_all = df[df["train_test"] == "test"]

    subj_list = np.unique(df["subject"])

    for test_subj in subj_list:
        # train_subjs = [subj for subj in subj_list if subj != test_subj]
        # Use the held-out subject for test (and validation??)

        if not os.path.exists(f'./{test_subj}_SHAP.pkl'):

            held_data = df[df["subject"] == test_subj]
            test = held_data[held_data["train_test"] == "test"]

            # Use all other subjects for train
            train_data = df[df["subject"] != test_subj]
            train_all = train_data[train_data["train_test"] != "test"]

            train, est = train_test_split(train_all, test_size=0.2)
            est, val = train_test_split(est, test_size=0.5)
            # Have: train, est, val, and test

            # Need: X_train, y_train, X_val, y_val, X_test, y_test

            # select only the features
            X_train = train.iloc[:, 3:184]
            X_est = est.iloc[:, 3:184]
            X_val = val.iloc[:, 3:184]
            X_test = test.iloc[:, 3:184]

            y_train = [0 if i == "READ" else 1 for i in train["labels"]]
            y_est = [0 if i == "READ" else 1 for i in est["labels"]]
            y_val = [0 if i == "READ" else 1 for i in val["labels"]]
            y_test = [0 if i == "READ" else 1 for i in test["labels"]]

            X_train = torch.tensor(X_train.values, dtype=torch.float32)
            X_est = torch.tensor(X_est.values, dtype=torch.float32)
            X_val = torch.tensor(X_val.values, dtype=torch.float32)
            X_test = torch.tensor(X_test.values, dtype=torch.float32)

            y_train = torch.tensor(y_train)
            y_est = torch.tensor(y_est)
            y_val = torch.tensor(y_val)
            y_test = torch.tensor(y_test)

            # Have: X_train, y_train, X_val, y_val, X_test, y_test

            # Create the model
            svc = SVC(kernel='rbf', C=1.0, class_weight='balanced')
            # Train the model
            svc.fit(X_train, y_train)
            # Evaluate the classifier

            print(f"Test Report: {test_subj}")
            predictions = svc.predict(X_test)

            report = classification_report(y_test, predictions)

            # Generate a conditional coalition function
            conditional_extension = weightedSHAP.generate_coalition_function(svc, X_train, X_est, 'classification', 'eeg')

            # With the conditional coalition function, compute attributions
            exp_dict = weightedSHAP.compute_attributions('classification', 'eeg',
                                                         svc, conditional_extension,
                                                         X_train, y_train,
                                                         X_val, y_val,
                                                         X_test, y_test)
            with open(f'./{test_subj}_perf.txt', 'w') as handle:
                handle.write(report)

            with open(f'./{test_subj}_SHAP.pkl', 'wb') as handle:
                pickle.dump(exp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print(exp_dict)

            if debug:
                break

