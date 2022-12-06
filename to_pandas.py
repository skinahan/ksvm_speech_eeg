# Create a python function which fulfills the following description.
# The function should load a series of numpy array datasets into a single pandas dataframe.
# The numpy arrays of the features are stored in the following file format:
# {subject}_{train/test}_compressed.npy
# The numpy arrays of the labels are stored in the following file format:
# {subject}_{train/test}_labels.npy
# The final pandas dataframe should have the following format:
# subject
# train/test
# multiple feature columns
# labels
import pandas as pd
import numpy as np
import os


def load_data(path):
    """
    Loads the data from the given path.
    :param path: The path to the data.
    :return: A pandas dataframe containing the data.
    """
    # Create a list of all the files in the given path.
    # Create a list of all the subjects.
    subjects = []
    # Create a list of all the train/test labels.
    train_test = []
    # Create a list of all the features.
    feats = [[] for i in range(180)]
    # Create a list of all the labels.
    labels = []
    files = os.listdir(path)
    for file in files:
        if file.endswith('_compressed.npy'):
            split_file = file.split('_')
            subj = split_file[0]
            file_train_test = split_file[1]
            # dim: 50 (trials) x 180 (features)
            # Need: to put each feature in its own array (so they can become pandas columns..)
            file_feats = np.load(path + file)
            """
            f4_feats = np.load(path + f'{subj}_{file_train_test}_f4s.npy')
            f5_feats = np.load(path + f'{subj}_{file_train_test}_f5s.npy')
            all_feats = np.append(file_feats, f4_feats)
            all_feats = np.append(all_feats, f5_feats)
            """
            file_feats_transp = np.transpose(file_feats)
            for idx, feat_col in enumerate(file_feats_transp):
                print(idx)
                feats[idx].extend(feat_col)
            file_labels = np.load(f'./labels/{subj}_{file_train_test}_labels.npy')
            file_labels = [label.replace(" ", "") for label in file_labels]
            subjects.extend([subj for feat in file_feats])
            train_test.extend([file_train_test for feat in file_feats])
            labels.extend(file_labels)
            #feats.append(file_feats)
    # Create a pandas dataframe containing
    # the subject, train/test, features and labels.
    df = pd.DataFrame({'subject': subjects,
                       'train_test': train_test,
                       'labels': labels})
    feat_ctr = 0
    for feat_arr in feats:
        feat_name = f'F{feat_ctr}'
        df[feat_name] = feat_arr
        feat_ctr += 1
    print(df)
    return df


if __name__ == '__main__':
    # Create the pandas dataset
    df = load_data('./data/')
    # Pickle it
    df.to_pickle('./eeg_dataset.pkl')
    # Load pickled pandas object from file
    df = pd.read_pickle('./eeg_dataset.pkl')
    print(df)
