'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2022/5/13 14:57
@Author : Qiufen.Chen
@FileName: evaluation.py
@Software: PyCharm
'''

from sklearn import metrics


def ACC(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: accuracy
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    acc = metrics.accuracy_score(y_true, y_pred)
    return acc


def F1_Score(y_true, y_pred):
    """
    The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are equal.

    :param y_pred: prediction value
    :param y_true: true value
    :return: F1 score
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    f1_score = metrics.f1_score(y_true, y_pred)
    return f1_score


def Precision(y_true, y_pred):
    """
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
    The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    :param y_pred: prediction value
    :param y_true: true value
    :return: precision
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    precision = metrics.precision_score(y_true, y_pred)
    return precision


def Recall(y_true, y_pred):
    """
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. 
    The recall is intuitively the ability of the classifier to find all the positive samples.
    :param y_pred: prediction value
    :param y_true: true value
    :return: recall
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    recall = metrics.recall_score(y_true, y_pred)
    return recall


def MCC(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: the Matthews correlation coefficient (MCC)
    """
    # y_true = y_true.detach().numpy()
    # y_pred = y_pred.detach().numpy()
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    return mcc


def AUC(y_true, y_pred):
    """
    :param y_pred: prediction value
    :param y_true: true value
    :return: the area under the ROC-curve (AUC)
    """
    roc = metrics.roc_auc_score(y_true, y_pred)
    return roc