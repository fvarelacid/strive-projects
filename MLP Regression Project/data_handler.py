import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch

def build_dataset(pth, batch_size, shuffle=True):
    #Fetch data from csv file pth
    data = pd.read_csv(pth)

    #####Preprocess data into numerical values#####
    #Transform Obejct data into categorical

    data['sex'] = data['sex'].astype('category')
    data['smoker'] = data['smoker'].astype('category')
    data['region'] = data['region'].astype('category')

    #Select the columns that are only categorical

    cat_columns = data.select_dtypes(['category']).columns

    #Apply cat.code on each column to transform it into numerical data

    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

    #Seperate data from target
    X = data.values[:, :-1]
    y = data.values[:, -1]

    #Split data into train and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Tranform training and testing data into tensors
    X_train = torch.tensor(X_train.astype(np.float32)).to(device)
    X_test = torch.tensor(X_test.astype(np.float32)).to(device)

    y_train = torch.tensor(y_train.astype(np.float32)).to(device)
    y_test = torch.tensor(y_test.astype(np.float32)).to(device)

    return X_train, X_test, y_train, y_test


def to_batches(X_train, X_test, y_train, y_test, batch_size, shuffle=True):

    # Gets the lenght of the table, shuffles it and return a list of the shuffled indexes
    train_indexes = np.random.permutation(X_train.shape[0])
    test_indexes = np.random.permutation(X_test.shape[0])

    # Arranges the X and y according to the new indexes order
    X_train = X_train[train_indexes]
    y_train = y_train[train_indexes]
    X_test = X_test[test_indexes]
    y_test = y_test[test_indexes]
    
    # Number of batches to be created - rounding to the closest integer
    n_batches = X_train.shape[0] // batch_size
    n_batches_test = X_test.shape[0] // batch_size

    # Creates X and y according to the new shape - using np.reshape
    X_train = X_train[:n_batches * batch_size].reshape(n_batches, batch_size, X_train.shape[1])
    y_train = y_train[:n_batches * batch_size].reshape(n_batches, batch_size, 1)
    X_test = X_test[:n_batches_test * batch_size].reshape(n_batches_test, batch_size, X_test.shape[1])
    y_test = y_test[:n_batches_test * batch_size].reshape(n_batches_test, batch_size, 1)

    return X_train, X_test, y_train, y_test