'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2023/3/15 0:15
@Author : Qiufen.Chen
@FileName: scaffold.py
@Software: PyCharm
'''

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, smiles in enumerate(dataset):
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds = []
    valid_inds = []
    test_inds = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("This path is exit!")


InputFile = "dataset/Cui9943.csv"
SavePath = "dataset/Cui/"

df = pd.read_csv(InputFile, header=0, sep=',', dtype=str)
smiles = df["SMILES"].tolist()
labels = df["LogS"].tolist()

train_inds, valid_inds, test_inds = scaffold_split(smiles, 0.1, 0.1)
name = ["smiles", "LogS"]
TrainData = [[smiles[idx], float(labels[idx])] for idx in train_inds]
ValData = [[smiles[idx], float(labels[idx])] for idx in valid_inds]
TestData = [[smiles[idx], float(labels[idx])] for idx in test_inds]

TrainData_df = pd.DataFrame(TrainData, columns=name)
TestData_df = pd.DataFrame(TestData, columns=name)

mkdir(SavePath)
TrainData_df.to_csv(SavePath + "train11.csv", index=False)
TestData_df.to_csv(SavePath + "test11.csv", index=False)
