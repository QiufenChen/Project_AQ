import torch as th
import mtMolDes

num_features = 4
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
feature_str = "h"
data_fn = "./dataset/Cui9943.csv"
#data_fn = "dataset/mini.csv" # This is a small data set (1000) for fast test.

data = mtMolDes.LoadGaoData(data_fn, num_features, feature_str, device)
solubNet = mtMolDes.PropPredBlock(num_features, feature_str)

training_ratio = 0.8
batch_size = 64
learning_rate = 1.e-3
max_epochs = 5
output_freq = 1
save_fn_prefix = "models/solub"

mtMolDes.Train(solubNet, data, training_ratio, learning_rate, batch_size, max_epochs, output_freq, save_fn_prefix, device)

for i,gx in enumerate(data):
    true_prop = gx[1]
    pred_prop = solubNet(gx[0])
    print("%5d %15.8f %15.8f" % (i, true_prop, pred_prop))
