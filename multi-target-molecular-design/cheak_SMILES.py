'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2023/3/9 17:14
@Author : Qiufen.Chen
@FileName: cheak_SMILES.py
@Software: PyCharm
'''

import pandas as pd
import numpy as np
import torch
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc, AllChem, rdMolDescriptors

# csv_file = "./extended_dataset/Boobier_100.csv"
# csv_file = "./extended_dataset/Lovric_829.csv"
# csv_file = "./extended_dataset/Gao_20.csv"
# csv_file = "./extended_dataset/CuiNovel_62.csv"
# csv_file = "./extended_dataset/Delaney_1128.csv"
# csv_file = "./extended_dataset/Llinas_132.csv"
# csv_file = "./extended_dataset/Llinas_100.csv"
csv_file = "extended_dataset/Llinas_32.csv"
df = pd.read_csv(csv_file, header=0, encoding="gbk")
smis = df['smiles']


#辅助函数
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


for item in smis:
    mol = Chem.MolFromSmiles(item)
    if mol is None:
        raise ValueError("Invalid SMILES code: %s" % (item))

    features = rdDesc.GetFeatureInvariants(mol)

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

    graph.ndata['x'] = torch.from_numpy(np.array(node_features))  # dgl添加原子/节点特征
    # G.edata['w'] = torch.from_numpy(np.array(edge_features))  # dgl添加键/边特征
    print(graph)

    #

