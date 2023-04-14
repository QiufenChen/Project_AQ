import torch
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
import time, os, csv
from .Evaluation import PCC, MAE, MSE, RMSE
import numpy as np
import random

import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing


# ===========================================================================================
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
# ===========================================================================================

def one_of_k_encoding_unk(x, allowable_set):
    '将x与allowable_set逐个比较，相同为True， 不同为False, 都不同则认为是最后一个相同'
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(tpsa, crippenlogPs, crippenMRs, LaASA):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    atom_features = [tpsa, crippenlogPs, crippenMRs, LaASA]
    return np.array(atom_features)


def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)


def ParseSMILES(sml, num_features, feature_str, device):
    """Transform a SMILES code to RDKIT molecule and DGL graph.

    Args:
        sml (str):          The SMILES code.
        num_features (int): The dimension of features for all atoms.
        feature_str (str):  The string to access the node features.
        device (str):       The device (CPU or GPU) to store the DGL graph.

    Returns:
        (mol, graph): The RDKIT molecule and DGL graph.
    """
    mol = Chem.MolFromSmiles(sml)
    if mol is None:
        raise ValueError("Invalid SMILES code: %s" % (sml))

    # features = rdDesc.GetFeatureInvariants(mol)

    # Calculation of the properties.
    AllChem.ComputeGasteigerCharges(mol)
    (CrippenlogPs, CrippenMRs) = zip(*(Chem.rdMolDescriptors._CalcCrippenContribs(mol)))
    TPSAs = Chem.rdMolDescriptors._CalcTPSAContribs(mol)
    (LaASAs, x) = Chem.rdMolDescriptors._CalcLabuteASAContribs(mol)

    graph = dgl.DGLGraph()

    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    graph.add_nodes(mol.GetNumAtoms())  # 添加节点
    node_features = []
    edge_features = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(TPSAs[i], CrippenlogPs[i], CrippenMRs[i], LaASAs[i])
        node_features.append(atom_i_features)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                graph.add_edges(i, j)
                # bond_features_ij = get_bond_features(bond_ij)
                # edge_features.append(bond_features_ij)

    graph.ndata['h'] = torch.from_numpy(np.array(node_features))  # dgl添加原子/节点特征
    # G.edata['w'] = torch.from_numpy(np.array(edge_features))  # dgl添加键/边特征
    return graph


def LoadGaoData(fn, num_features, feature_str, device):
    """Load data contributed by Dr. Peng Gao.

    Args:
        fn (str):           The file name.
        num_features (int): The dimension of features for all atoms.
        feature_str (str):  The string to access the node features.
        device (str):       The device (CPU or GPU) to store the DGL graph.

    Returns:
        [(graph, property)]: The DGL graph and property.
    """
    print("Load GaoDataSet from %s ... " % (fn), flush=True, end="\n")
    t0 = time.time()
    csv_reader = csv.reader(open(fn))
    next(csv_reader)
    data = []
    for line in csv_reader:
        graph = ParseSMILES(line[2], num_features, feature_str, device)
        prop = float(line[3])
        data.append([line[2], graph, prop])

    t1 = time.time()
    dur = t1 - t0
    print("done (%d lines, %.3f seconds) " % (len(data) + 1, dur), flush=True)
    return data


def Train(net, data, training_ratio, learning_rate, batch_size, max_epochs, output_freq, save_fn_prefix, device):
    """Train the net. The models will be saved.

    Args:
        net (pytorch module):       The net to train.
        data ([(graph, property)]): The data set.
        training_ratio (float):     The ratio of training data.
        learning_rate (float):      The learning rate for optimization.
        batch_size (int):           The batch size.
        max_epochs (int):           The number of epochs to train.
        output_freq (int):          The frequency of output.
        save_fn_prefix (str):       The net will save as save_fn_prefix+".pkl".
        device (str):               The device (CPU or GPU) to store the DGL graph.
    """
    # Prepare data.
    net.to(device)

    # 设置随机种子，结果可重复
    random.seed(1023)
    random.shuffle(data)

    num_data = len(data)
    num_training_data = int(num_data * training_ratio)
    training_data = data[:num_training_data]
    test_data = data[num_training_data:]
    num_test_data = len(test_data)
    training_graphs = [gx[1] for gx in training_data]
    training_labels = th.tensor([gx[2] for gx in training_data]).to(device)
    test_graphs = [gx[1] for gx in test_data]
    test_labels = th.tensor([gx[2] for gx in test_data]).to(device)

    # ==============================================================================

    # =================================== optimizer ================================
    optimizer = th.optim.RMSprop(net.parameters(), lr=learning_rate, alpha=0.9)
    # optimizer = th.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1.e-4)
    # optimizer = th.optim.ASGD(net.parameters(), lr=learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    # ==============================================================================

    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=10, threshold=0.0000001,
        threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)

    # criterion = th.nn.MSELoss(reduction='mean')   # MSE
    # criterion = th.nn.L1Loss(reduction='mean')  # MAE
    criterion = th.nn.SmoothL1Loss(reduction='mean')  # SmoothL1Loss

    # A closure to calculate loss.
    def getY(gs, ps):
        num_ps = ps.shape[0]
        p0s = th.zeros(num_ps).to(device)  # The predicted properties.
        for i in range(num_ps):
            p0s[i] = net(gs[i])
        return p0s, ps  # The predicted and true properties.

    # Set mini-batch.
    batch_idx = None
    if batch_size >= num_training_data:
        batch_idx = [[0, num_training_data]]
    else:
        batch_idx = [[i * batch_size, (i + 1) * batch_size] for i in range(num_training_data // batch_size)]
        if batch_idx[-1][1] != num_training_data: batch_idx.append([batch_idx[-1][1], num_training_data])

    # Output.
    print(">>> Training of the Model >>>")
    print("Start at: ", time.asctime(time.localtime(time.time())))
    print("PID:      ", os.getpid())
    print("# of all graphs/labels:      %d" % (num_data))
    print("# of training graphs/labels: %d" % (num_training_data))
    print("# of test graphs/labels:     %d" % (num_test_data))
    print("Learning rate:               %4.E" % (learning_rate))
    print("Batch size:                  %d" % (batch_size))
    print("Maximum epochs:              %d" % (max_epochs))
    print("Output frequency:            %d" % (output_freq))
    print("Params filename prefix:      %s" % (save_fn_prefix))
    print("Device:                      %s" % (device))
    separator = "-" * 150
    print(separator)
    print("%10s %15s %15s %15s %15s %15s %15s %15s %s" %
          ("Epoch", "TrainingLoss", "TrainMAE", "TrainMSE",
           "TestLoss", "TestMAE", "TestMSE", "Time(s)", "SavePrefix"))
    print(separator)

    # Training begins.
    t_begin = time.time()
    t0 = t_begin

    net.train()
    w_loss, w_mae, w_mse, w_pcc, w_rmse = 0, 0, 0, 0, 0
    for epoch in range(max_epochs + 1):
        # Do mini-batch.
        n = len(batch_idx)
        for idx in batch_idx:
            idx0 = idx[0]
            idx1 = idx[1]

            y_pred, y_ture = getY(training_graphs[idx0:idx1], training_labels[idx0:idx1])

            # print(th.any(th.isnan(y_pred)),th.any(th.isnan(y_ture)))

            # Calculate loss and other evaluation indicators.
            train_loss = criterion(y_pred, y_ture)
            mae = MAE(y_pred, y_ture)
            mse = MSE(y_pred, y_ture)
            rmse = RMSE(y_pred, y_ture)

            # Move forward.
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            train_loss.backward()
            optimizer.step()

            w_loss += train_loss.detach().item()
            w_mae += mae.detach().item()
            w_mse += mse.detach().item()
            w_rmse += rmse

        w_loss /= n
        w_mae /= n
        w_mse /= n
        w_rmse /= n

        scheduler.step(w_loss)

        # print("Epoch {:2d}, trian_loss: {:.4f}, trian_mae: {:.4f}, "
        #       "trian_mse:{:.4f}, trian_rmse: {:.4f}, trian_pcc: {:.4f}"
        #       .format(epoch, w_loss, w_mae, w_mse, w_rmse, w_pcc))


        # Output.
        if epoch % output_freq == 0:
            net.eval()
            y_pred, y_ture = getY(test_graphs, test_labels)

            test_loss = criterion(y_pred, y_ture)
            test_mae = MAE(y_pred, y_ture)
            test_mse = MSE(y_pred, y_ture)
            test_rmse = RMSE(y_pred, y_ture)

            net.train()
            prefix = save_fn_prefix + "-" + str(epoch)
            net.save(prefix)
            t1 = time.time()
            dur = t1 - t0
            print("%10d %15.7f %15.7f %15.7f %15.7f %15.7f %15.7f %15.7f  %s" % (epoch, w_loss, w_mae,
                                                                                 w_mse, test_loss, test_mae,
                                                                                 test_mse, dur, prefix), flush=True)

            # print("Epoch {:2d}, test_loss: {:.4f}, test_mae: {:.4f}, "
            #       "test_mse:{:.4f}, test_rmse: {:.4f}, test_pcc: {:.4f}"
            #       .format(epoch, test_loss, test_mae, test_mse, test_rmse, test_pcc))

            t0 = t1
    th.save(net.state_dict(), 'TAGCN.pt')
    t_end = time.time()
    print(separator)

    print("Final loss: %.4f, Final mse: %.4f, Final rmse: %.4f" % (w_loss, w_mse, w_rmse))
    print("Total training time: %.4f seconds" % (t_end - t_begin))
    print(">>> Training of the Model Accomplished! >>>")
    print("End at: ", time.asctime(time.localtime(time.time())))
