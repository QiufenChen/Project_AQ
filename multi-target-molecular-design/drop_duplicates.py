'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2023/3/12 22:19
@Author : Qiufen.Chen
@FileName: drop_duplicates.py
@Software: PyCharm
'''


import os
import pandas as pd

CuiFile = "dataset/Cui9943.csv"
Cui = pd.read_csv(CuiFile, header=0, sep=',')["SMILES"].tolist()
AqSolFile = "dataset/AqSol8048.csv"
AqSol = pd.read_csv(AqSolFile, header=0, sep=',')["SMILES"].tolist()


InputDir = "extended_dataset/"
SaveDir = "Independent/"
for Root, DirNames, FileNames in os.walk(InputDir):
    for FileName in FileNames:
        Name = FileName.split('.')[0].split('_')[0]
        FilePath = os.path.join(Root, FileName)

        InvalidID = []
        df = pd.read_csv(FilePath, header=0, sep=',')
        for i in range(len(df)):
            smile = df.loc[i]['smiles']
            if smile in Cui or smile in AqSol:
                InvalidID.append(i)
        df = df.drop(labels=InvalidID, axis=0)
        df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)  #删除全部为空的行
        df = df.reset_index(drop=True)

        # length = len(data)
        # print(length)
        # if len(data[0]) > 2:
        #     df = pd.DataFrame(data, columns=["smiles", "LogS", "Name"])
        # else:
        #     df = pd.DataFrame(data, columns=["smiles", "LogS"])
        df.to_csv(SaveDir+Name+str(len(df))+'.csv', encoding='utf-8', index=False)










