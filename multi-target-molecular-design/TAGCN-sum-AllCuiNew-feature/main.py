import random
import pandas as pd
import torch as th
from mtMolDes import model, Utility
from mtMolDes.Evaluation import MAE, MSE, RMSE, Spearman
from sklearn.metrics import r2_score
import numpy as np

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True
setup_seed(19970514)


num_features = 4
num_labels = 1
feature_str = "h"

batch_size = 32
learning_rate = 1e-3
max_epochs = 500
output_freq = 1

# device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
device = th.device("cpu")

data_file = "dataset/Cui_9943.csv"
all_data = Utility.LoadGaoData(data_file, num_features, feature_str, device)
print("# of all graphs/labels:      %d" % (len(all_data)))

solubNet = model.GCNNet(num_features, num_labels, feature_str)
Utility.Train(solubNet, all_data, learning_rate, batch_size, max_epochs, output_freq, device)
