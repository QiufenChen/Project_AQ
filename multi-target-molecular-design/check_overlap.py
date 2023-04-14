'''
# !/usr/bin/python3
# -*- coding: utf-8 -*-
@Time : 2023/3/9 22:50
@Author : Qiufen.Chen
@FileName: check_overlap.py
@Software: PyCharm
'''
import os

import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib_venn import venn2,venn2_circles
from rdkit import Chem


def Overlap(Name, Li):
    cui_file = "dataset/Cui9943.csv"
    cui = pd.read_csv(cui_file, header=0, sep=',')["SMILES"].tolist()
    AqSol_file = "dataset/AqSol8048.csv"
    AqSol = pd.read_csv(AqSol_file, header=0, sep=',')["SMILES"].tolist()

    print(Name + "和Cui_9943有重叠的数据个数为: %d" % len(list(set(cui) & set(Li))))
    print(Name + "和AqSol有重叠的数据个数为: %d" % len(list(set(AqSol) & set(Li))))


# InputDir = "Independent/"
# SaveDir = "Statistics/"
# for Root, DirNames, FileNames in os.walk(InputDir):
#     for FileName in FileNames:
#         Name = FileName.split('.')[0].split('_')[0]
#         FilePath = os.path.join(Root, FileName)
#
#         df = pd.read_csv(FilePath, header=0, sep=',')["LogS"]
#         plt.hist(df, color="r", alpha=0.5, bins=50, label=Name)
#         plt.legend(loc='upper left')
#         plt.savefig("./Statistics/" + Name + '.png', dpi=600)
#         plt.show()


Cui_file = "dataset/Cui/train.csv"
Cui = pd.read_csv(Cui_file, sep=",")["LogS"]
plt.hist(Cui,  bins=100, label="CuiTrain")
plt.legend(loc='upper left')
# plt.xlabel("logS")
# plt.ylabel("Frequency")
# plt.legend(loc='upper left')
# plt.savefig("./Statistics/" + 'AqSolTrain.png', dpi=600)
# plt.show()

AqSol_file = "dataset/Cui/test.csv"
AqSol = pd.read_csv(AqSol_file, header=0, sep=',')["LogS"]
plt.hist(AqSol, color="orange",bins=100, label="CuiTest")
plt.xlabel("logS")
plt.ylabel("Frequency")
plt.legend(loc='upper left')
plt.savefig("./Statistics/" + 'Cui.png', dpi=600)
plt.show()

# gao_file = "Independent/Gao20.csv"
# Delaney_file = "extended_dataset/Delaney_1128.csv"
# gao = pd.read_csv(gao_file, header=0, sep=',')["smiles"].tolist()
# Delaney = pd.read_csv(Delaney_file, header=0, sep=',')["smiles"].tolist()
# print("Gao和Delaney有重叠的数据个数为: %d" % len(list(set(gao) & set(Delaney))))

# plt.hist(gao, color="r", alpha=0.5, bins=100, label="Gao20")
# plt.legend(loc='upper left')
# plt.savefig("./Statistics/" + 'Gao20.png')
# plt.show()


#
# # print(len(set(AqSol) & set(Cui)))
# fig,axs=plt.subplots(1,3, figsize=(10,8),dpi=150)
# g=venn2(subsets = [{1,2,3},{1,2,4}],
#         set_labels = ('Label 1', 'Label 2'),
#         set_colors=("#098154","#c72e29"),
#         alpha=0.6,
#         normalize_to=1.0,
#         ax=axs[0],#该参数指定
#        )

# CuiInChIKey = cui_df['InChIKey'].tolist()
# AqSolInChIKey = AqSol_df['InChIKey'].tolist()
#
#
# non_Duplicates = []
# for index, item in enumerate(AqSolInChIKey):
#     if item not in CuiInChIKey:
#         sml = AqSol_df['SMILES'][index]
#         mol = Chem.MolFromSmiles(sml)
#         if mol is None:
#             raise ValueError("Invalid SMILES code: %s" % (sml))
#
#         non_Duplicates.append([AqSol_df['ID'][index],
#                                AqSol_df['InChIKey'][index],
#                                AqSol_df['SMILES'][index],
#                                AqSol_df['Solubility'][index]])
#
# print("AqSolDB数据库总共有 %d 条数据" % (len(AqSolInChIKey)))
# print("移除和Cui2020的重复数据 %d 条" % (len(AqSolInChIKey) - len(non_Duplicates)))
# print("剩余可用数据 %d 条" % (len(non_Duplicates)))
# data = pd.DataFrame(columns=['Compound ID', 'InChIKey', 'SMILES', 'logS'], data=non_Duplicates)  # 先变成df格式后变成二位数句
# data.to_csv('./dataset/' + "AqSol8048.csv", index=False, encoding='utf-8')
