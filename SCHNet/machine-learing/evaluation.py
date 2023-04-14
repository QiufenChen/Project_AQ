'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/5/13 14:57
@Author : Qiufen.Chen
@FileName: evaluation.py
@Software: PyCharm
'''
import itertools

from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

plt.rcParams['axes.linewidth'] = 1  # 图框宽度
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 20}



def ACC(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Pearson's correlation coefficient
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return accuracy


def Precision(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Root Mean Square Error
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    precision = metrics.precision_score(y_true, y_pred)
    return precision


def Recall(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Mean Absolute Error
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    recall = metrics.recall_score(y_true, y_pred)
    return recall


def F1_Score(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: coefficient of determination
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()

    p = Precision(y_true, y_pred)
    r = Recall(y_true, y_pred)
    f1_score = 2*p*r / (p+r+1e-10)
    return f1_score


def MCC(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: Mean Square Error
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    return mcc


def Confusion_Matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, path=""):
    """
    画混淆矩阵
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    画图函数 输入：
    cm 矩阵
    classes 输入str类型
    title 名字
    cmap [图的颜色设置](https://matplotlib.org/examples/color/colormaps_reference.html)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=(11, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, font)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # plt.gca().set_xticks(tick_marks, minor=True)
    # plt.gca().set_yticks(tick_marks, minor=True)
    # plt.gca().xaxis.set_ticks_position('none')
    # plt.gca().yaxis.set_ticks_position('none')
    # plt.grid()
    # plt.gcf().subplots_adjust(bottom=0.1)
    # plt.tight_layout()
    plt.ylabel('True label', font)
    plt.xlabel('Predicted label', font)
    # 解决中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.savefig(path, dpi=500)

