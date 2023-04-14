# Author QFIUNE
# coding=utf-8
# @Time: 2023/3/16 21:48
# @File: TenFold.py
# @Software: PyCharm
# @contact: 1760812842@qq.com


import collections
import copy
import os
import time

import random
import pandas as pd
import numpy as np
import scipy.stats
import torch as th
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit, StratifiedKFold
from mtMolDes import model, Utility
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

"""
Reference to: https://blog.csdn.net/weixin_43646592/article/details/121365173
"""
separator = "-" * 100

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
setup_seed(42)


def getY(gs, ps, net):
    num_ps = ps.shape[0]
    p0s = th.zeros(num_ps)
    for i in range(num_ps):
        p0s[i] = th.sum(net(gs[i]), dim=0)
    return p0s, ps  


def myDataLoader(batch_size, num):
    """
    Set mini-batch.
    :param batch_size:
    :param num:
    :return:
    """
    batch_idx = None
    if batch_size >= num:
        batch_idx = [[0, num]]
    else:
        batch_idx = [[i * batch_size, (i + 1) * batch_size] for i in range(num // batch_size)]
        if batch_idx[-1][1] != num: batch_idx.append([batch_idx[-1][1], num])
    return batch_idx


def criterionR(output, target):
    target_mean = th.mean(target)
    ss_tot = th.sum((target - target_mean) ** 2)
    ss_res = th.sum((target - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def Train(net, train_graphs, train_labels, learning_rate, batch_size, max_epochs, fold):
    """
    Train the net. The models will be saved.
    """
    optimizer = th.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1.e-4)
    # optimizer = th.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9, weight_decay=1.e-4)
    # scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.9, patience=10, threshold=0.0000001,
    #     threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=3, threshold=0.0001,
        threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)
   
    # criterion = th.nn.SmoothL1Loss(reduction='mean')  # SmoothL1Loss
    # criterion =  th.nn.L1Loss(reduction='mean')
    criterionL = th.nn.MSELoss(reduction='mean')     # MSE

    # print(">>> Training of the Model >>>")
    # print("Start at: ", time.asctime(time.localtime(time.time())))
    # print("PID:      ", os.getpid())

    net.train()
    batch_idx = myDataLoader(batch_size, len(train_graphs))

    # Training begins.
    t_begin = time.time()
    t0 = t_begin

    # MinValMae = 10
    # BestTrainMAE, BestTrainRMSE, BestTrainR2, BestTrainCC = 0, 0, 0, 0
    # BestTestMAE, BestTestRMSE, BestTestR2, BestTestCC = 0, 0, 0, 0
    # BestEpoch = 0
    # BestModel = None

    train_epochs_loss = []
    valid_epochs_loss = []

    net.train()
    for epoch in range(max_epochs + 1):
        # n = len(batch_idx)
        # w_loss, w_mae, w_rmse, w_r2, w_cc = 0, 0, 0, 0, 0
        train_epoch_loss = []
        train_epoch_pred = []
        train_epoch_true = []

        for idx in batch_idx:
            idx0 = idx[0]
            idx1 = idx[1]
            # print(train_graphs[idx0:idx1], train_labels[idx0:idx1])
            y_pred, y_true = getY(train_graphs, th.tensor(train_labels), net)
            loss = th.sqrt(criterionL(y_pred, y_true)) - 0.1 * criterionR(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.item())
            y_pred = [i.item() for i in y_pred]
            y_true = [i.item() for i in y_true]
            train_epoch_pred += y_pred
            train_epoch_true += y_true
        # print(train_epoch_true)
        y_test, y_predict = np.array(train_epoch_true), np.array(train_epoch_pred)

        train_loss = np.average(train_epoch_loss)
        # train_epochs_loss.append(train_loss)

        train_mae = mean_absolute_error(y_test, y_predict)
        train_rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        train_r2 = r2_score(y_test, y_predict)
        train_cc = scipy.stats.spearmanr(y_test, y_predict)[0]

        scheduler.step(train_loss)
        print("%10d %15.3f %15.3f %15.3f %15.3f %15.3f " %(epoch, train_loss, train_mae, train_rmse, train_r2, train_cc))
    th.save(net.state_dict(), './models/'+'solubNet'+str(fold+1)+'.pt')
    

def Eval(net, test_graphs, test_labels):
    net.eval()
    with th.no_grad():
        y_pred, y_true = getY(test_graphs, th.tensor(test_labels), net)
        y_pred = np.array([i.item() for i in y_pred])
        y_true = np.array([i.item() for i in y_true])
        test_mae = mean_absolute_error(y_true, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        test_r2 = r2_score(y_true, y_pred)
        test_cc = scipy.stats.spearmanr(y_true, y_pred)[0]

        return test_mae, test_rmse, test_r2, test_cc


history = collections.defaultdict(list)
skf = KFold(n_splits=5, shuffle=True, random_state=42)

learning_rate = 0.001
batch_size = 32
max_epochs = 500
num_features = 4
num_labels = 1
feature_str = "h"
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
InputFile = "./dataset/Cui/train.csv"
data = Utility.LoadGaoData(InputFile, num_features, feature_str, device)
graphs = [gx[1] for gx in data]
labels = th.tensor([gx[2] for gx in data])

for fold, (train_idx, val_idx) in enumerate(skf.split(graphs, labels)):
    print('**' * 10, fold + 1, ' fold ing....', '**' * 10)
    x_train = [graphs[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]

    x_val = [graphs[i] for i in val_idx]
    y_val = [labels[i] for i in val_idx]

    solubNet = model.GCNNet(num_features, num_labels, feature_str)
    print(separator)
    print("%10s %15s %15s %15s %15s %15s" % ("Epoch", "TrainLoss", "TrainMAE", "TrainRMSE", "TrainR2", "TrainCC"))
    print(separator)
    Train(solubNet, x_train, y_train, learning_rate, batch_size, max_epochs, fold)
    print(separator)
    print("%15s %15s %15s %15s" % ("TestMAE", "TestRMSE", "TestR2", "TestCC"))
    print(separator)

    MAE,RMSE,R2,PCC = Eval(solubNet, x_val, y_val)
    print("%15.3f %15.3f %15.3f %15.3f " % (MAE,RMSE,R2,PCC))
    history['MAE'].append(MAE)
    history['RMSE'].append(RMSE)
    history['R2'].append(R2)
    history['PCC'].append(PCC)


MeanMAE = np.mean(history['MAE'])
MeanRMSE = np.mean(history['RMSE'])
MeanR2 = np.mean(history['R2'])
MeanPCC = np.mean(history['PCC'])
print(separator)
print("%15s %15s %15s %15s" % ("MeanMAE", "MeanRMSE", "MeanR2", "MeanCC"))
print(separator)
print("%15.3f %15.3f %15.3f %15.3f " % (MeanMAE,MeanRMSE,MeanR2,MeanPCC))
