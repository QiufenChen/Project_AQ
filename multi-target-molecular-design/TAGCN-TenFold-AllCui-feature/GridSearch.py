# Author QFIUNE
# coding=utf-8
# @Time: 2023/3/17 11:17
# @File: GridSearch.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
from torch import optim
import torch as th

from mtMolDes import model, Utility

"""Reference to: https://blog.csdn.net/deephub/article/details/128981308"""

num_features = 4
num_labels = 1
feature_str = "h"
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
solubNet = model.GCNNet(num_features, num_labels, feature_str)
TrainFile = "dataset/Cui/train.csv"
data_train = Utility.LoadGaoData(TrainFile, num_features, feature_str, device)
train_graphs = [gx[1] for gx in data_train]
train_labels = th.tensor([gx[2] for gx in data_train]).to(device)

Net = NeuralNetRegressor(
    module=solubNet,
    criterion=nn.SmoothL1Loss(),
    optimizer=optim.Adam,
    verbose=False)

# define the grid search parameters
param_grid = {
 'batch_size': [8, 16, 32, 64],
 'max_epochs': [100, 200, 300, 400, 500]
}

grid = GridSearchCV(estimator=Net, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(th.sum(solubNet(train_graphs[0]), dim=0), train_labels[0])

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
