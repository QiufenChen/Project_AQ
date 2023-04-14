'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2023/3/13 16:29
@Author : Qiufen.Chen
@FileName: test_soltrannet.py
@Software: PyCharm
'''
import os

"""
Code source: https://github.com/gnina/SolTranNet
pip install soltrannet
"""

import soltrannet as stn
import pandas as pd
from mtMolDes.Evaluation import Spearman
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from ultity import WriteLog


WriteLog.make_print_to_file(path='./')
InputDir = "TestSet/"
for Root, DirNames, FileNames in os.walk(InputDir):
    for FileName in FileNames:
        Name = FileName.split('.')[0]
        FilePath = os.path.join(Root, FileName)
        df = pd.read_csv(FilePath, header=0, sep=',')
        smiles = df["smiles"]
        y_true = df["LogS"].to_numpy()
        y_pred = np.array([item[0] for item in list(stn.predict(smiles))])

        mae = mean_absolute_error(y_pred, y_true)
        rmse = np.sqrt(mean_squared_error(y_pred, y_true))
        r2 = r2_score(y_pred, y_true)
        cc = Spearman(y_pred, y_true)

        separator = "-" * 70
        print(separator)
        print("This is a test on Dataset " + Name)
        print("%15s %15s %15s %15s" %
              ("TestMAE", "TestRMSE", "TestR2", "TestCC"))
        print("%15.3f %15.3f %15.3f %15.3f" % (mae, rmse, r2, cc))
        print(separator)





