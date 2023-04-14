# Author QFIUNE
# coding=utf-8
# @Time: 2022/4/24 9:34
# @File: draw.py
# @Software: PyCharm
# @contact: 1760812842@qq.com


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def readCSV(file):
    """
    purpose: get loss, mae, mse and rmse value
    :param file: csv file
    :return: list
    """
    df = pd.read_excel(file, usecols=[2, 4, 6, 8])
    data = np.array(df)
    data = data.tolist()
    return data


def readTxt(file):
    """
    purpose: get train loss and test loss
    :param file: log file, the suffix is .log.
    :return:
    """
    epoch = []
    train = []
    test = []
    with open(file, 'r') as fin:
        lines = fin.read()
        separator = "-" * 100
        paralist = lines.split(separator)

        lossdata = paralist[2]
        for item in lossdata.split('\n')[1:-1]:
            res = item.split()
            print(res)
            epoch.append(int(res[0]))
            train.append(float(res[1]))
            test.append(float(res[2]))
        return epoch, train, test


def Draw(epoch, train, test):
    plt.figure(figsize=(15, 8), dpi=300)
    x = epoch
    y1 = train
    y2 = test

    # # 设置Y轴标题
    # plt.ylabel("The trend of loss value")
    # plt.xlabel("Epoch")
    #
    # plt.tick_params(labelsize=8)
    # plt.plot(x, train_loss, 'blue', x, val_loss, 'orange')
    # plt.legend(labels=["Train Loss", "Validation Loss"])
    # plt.show()

    # # 设置Y轴标题
    # plt.ylabel("The trend of MAE value")
    # plt.xlabel("Epoch")
    #
    # plt.tick_params(labelsize=8)
    # plt.plot(x, train_mae, 'r--', x, val_mae, '-g')
    # plt.legend(labels=["Train MAE", "Validation MAE"])
    #
    # plt.show()

    # 设置Y轴标题
    plt.ylabel("The trend of MAE value")
    plt.xlabel("Epoch")

    plt.tick_params(labelsize=8)
    plt.plot(x, y1, '#8A2BE2', x, y2, '#66CDAA', linewidth=5.0)
    plt.legend(labels=["Train MAE", "Validation MAE"])

    plt.show()

file = 'D:/Project_QF/Molecular_properties/train_solub.log'
epoch, train, test = readTxt(file)
Draw(epoch, train, test)
