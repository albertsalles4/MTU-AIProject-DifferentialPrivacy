import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def train_test_split_sorted(features, labels, test_size=0.2):
    n = len(features)
    n_test = int(n * test_size)
    n_train = n - n_test
    X_train = features[:n_train]
    y_train = labels[:n_train]
    X_test = features[n_train:]
    y_test = labels[n_train:]
    return X_train, X_test, y_train, y_test


def load_data(csv_file=None, test_size=0.25, data_folder=None):
    if csv_file is None and data_folder is None:
        raise ValueError('You must provide a csv file or a data folder')
    

    if csv_file:
        # 1. Load and Prepare the Data  #
        data = pd.read_csv('data/creditcard.csv')
        data = data.sort_values('Time')

        # Extract features and labels
        features = data.drop(columns=['Class', 'Time']).values
        labels = data['Class'].values


        X_train, X_test, y_train, y_test = train_test_split_sorted(features, labels, test_size=test_size)

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save train and test indices to recreate the dataset
        np.savetxt('data/X_train.txt', X_train, fmt='%d')
        np.savetxt('data/y_train.txt', y_train, fmt='%d')
        np.savetxt('data/X_test.txt', X_test, fmt='%d')
        np.savetxt('data/y_test.txt', y_test, fmt='%d')

    if data_folder:
        # Load train and test indices
        X_train = np.loadtxt(os.path.join(data_folder, 'X_train.txt'))
        y_train = np.loadtxt(os.path.join(data_folder, 'y_train.txt'))
        X_test = np.loadtxt(os.path.join(data_folder, 'X_test.txt'))
        y_test = np.loadtxt(os.path.join(data_folder, 'y_test.txt'))

        scaler = StandardScaler().fit(X_train)

    return X_train, X_test, y_train, y_test, scaler