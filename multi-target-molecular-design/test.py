# Author QFIUNE
# coding=utf-8
# @Time: 2022/6/22 16:01
# @File: test.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

import os
import torch as th
import mtMolDes
from LRPExplanation import LRPModel
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions


def visualization(graph, R, sml):
    plt.figure(figsize=(15, 8))
    g = graph.to_networkx(node_attrs='h', edge_attrs='')  # 转换 dgl graph to networks
    pos = nx.kamada_kawai_layout(g)

    pos_higher = {}
    for k, v in pos.items():
        if (v[1] > 0):
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)

    nodes = []
    weights = []
    for idx_atom in range(graph.nodes().shape[0]):
        weight = float(R[idx_atom])
        weights.append(weight)
        nodes.append((idx_atom, {"weight": weight}))
    g.add_nodes_from(nodes)

    cmap = plt.cm.get_cmap('Greens')
    nx.draw(g, with_labels=True, pos=pos, node_color=weights, cmap=cmap, node_size=800, font_color="black")

    plt.show()
    plt.savefig('./graphs/' + sml + ".png", format="PNG")


# ======================================================================================================================
num_features = 1
feature_str = 'h'
EPSILON = 0.01
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
data_fn = "dataset/test_solub.csv"
data = mtMolDes.LoadGaoData(data_fn, num_features, feature_str, device)
solubNet = mtMolDes.PropPredBlock(num_features, feature_str)

project_path = os.getcwd()
model_path = project_path + '/models/TAGCN_0.001.pt'
solubNet.load_state_dict(th.load(model_path, map_location='cpu'))


print("load success")
print('-'*50)
for i, gx in enumerate(data):
    true_prop = gx[2]
    pred_prop = solubNet(gx[1])
    # print("%5d %15.8f %15.8f" % (i, true_prop, pred_prop))

    # Gain relevance of each layer
    print(" ---------- Explanation ----------")
    sml = data[i][0]
    graph = data[i][1]
    R = LRPModel(solubNet)(data[i][1])
    visualization(graph, R[-1], sml)

    mol = Chem.MolFromSmiles(sml)
    opts = DrawingOptions()
    opts.includeAtomNumbers = True
    opts.bondLineWidth = 2.8
    draw = Draw.MolToImage(mol, options=opts, size=(600, 600))
    draw.save('./Molecules/' + sml + '.jpg')

    Draw.MolToImage(mol, size=(600, 600), kekulize=True)
    Draw.ShowMol(mol, size=(600, 600), kekulize=False)
    Draw.MolToFile(mol, './Molecules/' + sml + '.png', size=(600, 600), kekulize=False)
