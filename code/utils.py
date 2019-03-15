#!/usr/bin/env python3

# Importing the libraries
import pandas as pd

def save_acc(train, test, output_file):

    tuples = []

    for l in train:
        for i in range(len(l)):
            tuples.append((i + 1, l[i], "train"))

    for l in test:
        for i in range(len(l)):
            tuples.append((i + 1, l[i], "test"))

    col_names = ("epochs", "accuracy", "type")

    frame = pd.DataFrame.from_records(tuples, columns=col_names)
    frame.to_csv(output_file + "_acc.csv", index=False, sep='\t', encoding='utf-8')


def save_loss(train, test, output_file):

    tuples = []

    for l in train:
        for i in range(len(l)):
            tuples.append((i + 1, l[i], "train"))

    for l in test:
        for i in range(len(l)):
            tuples.append((i + 1, l[i], "test"))

    col_names = ("epochs", "loss", "type")

    frame = pd.DataFrame.from_records(tuples, columns=col_names)
    frame.to_csv(output_file + "_loss.csv", index=False, sep='\t', encoding='utf-8')
