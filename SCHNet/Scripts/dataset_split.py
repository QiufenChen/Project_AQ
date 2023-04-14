import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem import AllChem


validation_split = .1
shuffle_dataset = True
random_seed = 42
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="dataset to train")
args = parser.parse_args()
args.dataset = 'E:/NextStage/5-Practice/data/processed_data/solub/known_solub'
dataset = args.dataset
save_path = 'E:/NextStage/Prediction-with-GCN/Aqueous-solubility-prediction-with-GCN-master/Dataset/solub/solub'

delaney = pd.read_csv(dataset + ".csv", skiprows=1,
                      names=['smiles', 'measured'])
dataset_size = len(delaney)
invalid_id = []
for i in range(dataset_size):
    smi = delaney.loc[i]['smiles']
    try:
        mol = Chem.MolFromSmiles(smi)
        AllChem.Compute2DCoords(mol)

    except:
        print(smi + "was not valid smiles\n")
        invalid_id.append(i)
delaney.drop(labels=invalid_id, axis=0)

# Creating data indices for training and validation splits:
dataset_size = len(delaney)
print('dataset_size= %s' % (dataset_size))
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# print(train_indices)

# Creating PT data samplers and loaders:
train_sampler = delaney.loc[train_indices]
train_mean = train_sampler['measured'].mean()
train_std = train_sampler['measured'].std()
train_sampler['measured'] = (
    train_sampler['measured'] - train_mean) / train_std

valid_sampler = delaney.loc[val_indices]
valid_sampler['measured'] = (valid_sampler['measured']-train_mean)/train_std

mean_file = open(save_path + '_mean_std.txt', 'w+')
mean_file.writelines('train_mean= %s\n' % (train_mean))
mean_file.writelines('train_std= %s' % (train_std))


train_sampler.to_csv(save_path +"_train.csv", index=False)
valid_sampler.to_csv(save_path +"_valid.csv", index=False)


