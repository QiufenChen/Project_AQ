# Author QFIUNE
# coding=utf-8
# @Time: 2022/7/6 10:46
# @File: pre_processing.py
# @Software: PyCharm
# @contact: 1760812842@qq.com
import csv

from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
import selfies as sf
import numpy as np

fn = './dataset/data4.csv'
save_path = './dataset/'

validation_split = .1
shuffle_dataset = True
random_seed = 42

delaney = pd.read_csv(fn, skiprows=1, names=['smiles', 'measured'])
dataset_size = len(delaney)
print(dataset_size)
invalid_id = []
for i in range(dataset_size):

    sml = delaney.loc[i]['smiles']
    try:
        new_sml = sf.decoder(sf.encoder(sml))
        mol = Chem.MolFromSmiles(new_sml)
        AllChem.Compute2DCoords(mol)
        print(sml)

    except:
        print(str(sml) + "was not valid smiles\n")
        invalid_id.append(i)

li = delaney.drop(labels=invalid_id, axis=0)
li = li.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) #删除全部为空的行


indices = list(range(len(li)))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
train_sampler = li.loc[indices]
# valid_sampler = li.loc[val_indices]

train_sampler.to_csv(save_path + "data4_preprocess.csv", index=False)
# valid_sampler.to_csv(save_path + "negative_test.csv", index=False)