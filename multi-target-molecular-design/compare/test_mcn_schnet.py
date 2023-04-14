# Author QFIUNE
# coding=utf-8
# @Time: 2023/3/14 9:35
# @File: test_mcn_schnet.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import pandas as pd
import numpy as np


# GaoFile = "Independent/CuiNovel62.csv"
GaoFile = "Gao.csv"
Gao = pd.read_csv(GaoFile, header=0, sep=',')
y_pred = Gao["D"].to_numpy()

# smiles = Gao["smiles"].tolist()
y_true = Gao["LogS"].to_numpy()
#
# pred = list(stn.predict(smiles))
#
# y_pred = []
# for item in pred:
#     y_pred.append(item[0])

y_pred = np.array(y_pred)