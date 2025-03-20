import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def train_test_split_sorted(features, labels, test_size=0.2):
    """Gets last test_size of positive samples and all the negative samples that come after them"""
    positive_idx = np.where(labels == 1)[0]
    percentile_idx = positive_idx[int(np.ceil(len(positive_idx) * (1 - test_size))) - 1]

    X_train = features[:percentile_idx]
    y_train = labels[:percentile_idx]
    X_test = features[percentile_idx:]
    y_test = labels[percentile_idx:]
    
    return X_train, X_test, y_train, y_test


def load_data(csv_file=None, test_size=0.2, val_size=0.1, data_folder=None):
    if csv_file is None and data_folder is None:
        raise ValueError('You must provide a csv file or a data folder')
    

    if csv_file:
        # 1. Load and Prepare the Data  #
        data = pd.read_csv(csv_file)
        data = data.sort_values('Time')

        # Extract features and labels
        features = data.drop(columns=['Class', 'Time']).values
        labels = data['Class'].values

        print('Dataset description:')
        print(f'Number of samples: {len(labels)}')
        print(f'Number of positive samples: {np.sum(labels)}')
        print(f'Number of negative samples: {len(labels) - np.sum(labels)}')


        X_train, X_test, y_train, y_test = train_test_split_sorted(features, labels, test_size=test_size)
        X_train, X_val, y_train, y_val = train_test_split_sorted(X_train, y_train, test_size=val_size)

        # Save train and test indices to recreate the dataset
        np.savetxt(os.path.join(data_folder, 'X_train.txt'), X_train, fmt='%d')
        np.savetxt(os.path.join(data_folder, 'y_train.txt'), y_train, fmt='%d')
        np.savetxt(os.path.join(data_folder, 'X_test.txt'), X_test, fmt='%d')
        np.savetxt(os.path.join(data_folder, 'y_test.txt'), y_test, fmt='%d')
        np.savetxt(os.path.join(data_folder, 'X_val.txt'), X_val, fmt='%d')
        np.savetxt(os.path.join(data_folder, 'y_val.txt'), y_val, fmt='%d')

    elif data_folder:
        # Load train and test indices
        X_train = np.loadtxt(os.path.join(data_folder, 'X_train.txt'))
        y_train = np.loadtxt(os.path.join(data_folder, 'y_train.txt'))
        X_test = np.loadtxt(os.path.join(data_folder, 'X_test.txt'))
        y_test = np.loadtxt(os.path.join(data_folder, 'y_test.txt'))
        X_val = np.loadtxt(os.path.join(data_folder, 'X_val.txt'))
        y_val = np.loadtxt(os.path.join(data_folder, 'y_val.txt'))

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_val, X_test, y_train, y_val, y_test