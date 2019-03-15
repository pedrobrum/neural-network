#!/usr/bin/env python3

# Importing the libraries
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import argparse

from sklearn import preprocessing, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.optimizers import SGD, Adam, RMSprop

from utils import save_acc, save_loss


def train_model(x_train, x_test, y_train, y_test, parameters, input_dim):

    # parameters
    batch_size = parameters["batch_size"]
    epochs = parameters["epochs"]
    lr = parameters['lr']
    hidden_units = parameters['hidden_units']
    output_units = parameters['output_units']
    more_layer = parameters['more_layer']

    # Linear strack of layers
    model = Sequential()

    # Adding the input layer and the first hidden layer
    model.add(Dense(units=hidden_units, activation='relu',
                    input_dim=input_dim))

    # Adding more hiddens layers
    if more_layer == 1:
        model.add(Dense(units=30, activation='relu'))

    # Adding the output layer
    model.add(Dense(units=output_units, activation='softmax'))

    sgd = SGD(lr=lr)

    # Compiling
    model.compile(optimizer=sgd, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Fitting model
    result = model.fit(x=x_train, y=y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True)

    return result.history


def kfold_model(x, y, parameters, input_dim, k=3):

    # applying cross-validation
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    acc_train = []
    acc_test = []
    loss_train = []
    loss_test = []

    for train, test in kfold.split(x, y):
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        result = train_model(x_train, x_test, y_train, y_test, parameters, input_dim)
        
        # saving accuracy
        acc_train.append(result['acc'])
        acc_test.append(result['val_acc'])

        # saving loss
        loss_train.append(result['loss'])
        loss_test.append(result['val_loss'])

    return acc_train, acc_test, loss_train, loss_test


def train_test_data(dataset):

    # drop irrelevant features
    dataset = dataset.drop(['objid', 'specobjid', 'fiberid', 'run', 'rerun', 'camcol', 'field'], axis=1)

    x = dataset.drop(['class'], axis=1)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = np.array(x)

    y = dataset['class']
    # Transform classes names into numerical values
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    return x, y


def balanced_dataset(x, y, parameters, input_dim):

    # applying oversampling
    balance = RandomOverSampler()

    x, y = balance.fit_sample(x, y)
    acc_train, acc_test, loss_train, loss_test = kfold_model(x, y, parameters, input_dim)

    return acc_train, acc_test, loss_train, loss_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="input file.")
    parser.add_argument("-o", "--output", required=True,
                        help="output file.")
    parser.add_argument("-b", "--batch_size", type=int, default=50,
                        help="batch size.")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="number of epochs to train the model.")
    parser.add_argument("-l", "--lr", type=float, default=0.1,
                        help="learning rate.")
    parser.add_argument("-x", "--hidden_units", type=int, default=50,
                        help="number of units in the hidden layers.")
    parser.add_argument("-y", "--output_units", type=int, default=3,
                        help="number of units in the output layer.")
    parser.add_argument("-m", "--more_layer", type=int, default=0,
                        help="add one more hidden layer.")
    parser.add_argument("-v", "--oversampling", type=int, default=0,
                        help="oversampling.")

    return parser.parse_args()


def main():

    args = parse_args()

    # read parameters
    input_file = args.input
    output_file = args.output

    parameters = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_units": args.hidden_units,
        "output_units": args.output_units,
        "more_layer": args.more_layer
    }

    dataset = pd.read_csv(input_file)
    x, y = train_test_data(dataset)

    input_dim = len(x[0])

    # using oversampling
    if args.oversampling == 1:
        acc_train, acc_test, loss_train, loss_test = balanced_dataset(x, y, parameters, input_dim)
    else:
        acc_train, acc_test, loss_train, loss_test = kfold_model(x, y, parameters, input_dim)
    
    # save_acc(acc_train, acc_test, output_file)
    # save_loss(loss_train, loss_test, output_file)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % ((time.time() - start_time)))

