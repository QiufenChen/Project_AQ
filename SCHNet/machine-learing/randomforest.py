# Author QFIUNE
# coding=utf-8
# @Time: 2022/8/8 19:55
# @File: randomforest.py
# @Software: PyCharm
# @contact: 1760812842@qq.com

# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
import joblib
import itertools
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings("ignore")


class Ctrain_forest:
    '''
    调用sklearn 实现Random Forest功能：
    画混淆矩阵
    输入数据实现训练
    保存模型到指定位置
    调用模型实现预测
    '''

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, path="rfc"):
        """
        画混淆矩阵
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        画图函数 输入：
        cm 矩阵
        classes 输入str类型
        title 名字
        cmap [图的颜色设置](https://matplotlib.org/examples/color/colormaps_reference.html)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        plt.figure(figsize=(11, 8))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        # plt.gca().set_xticks(tick_marks, minor=True)
        # plt.gca().set_yticks(tick_marks, minor=True)
        # plt.gca().xaxis.set_ticks_position('none')
        # plt.gca().yaxis.set_ticks_position('none')
        # plt.grid()
        # plt.gcf().subplots_adjust(bottom=0.1)
        # plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # 解决中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.savefig(path, dpi=500)
        # plt.show()

    def train_forest(self,x_train, y_train, x_test, y_test, path):
        """
        Random Foeset类
        输入：
        x、y以实现训练,path是保存训练过程的路径
        输出：
        clf 模型
        matrix 混淆矩阵
        dd classifi_report
        kappa kappa系数
        acc_1 模型精度
        """

        # 寻找最优参数
        depth = np.arange(1, 50, 1)
        acc_list = []
        for d in depth:
            clf = RandomForestClassifier(bootstrap=True, class_weight="balanced", criterion='gini',
                                         max_depth=d * 10 + 1, max_features='auto', max_leaf_nodes=None,
                                         min_impurity_decrease=0.0, min_impurity_split=None,
                                         min_samples_leaf=3, min_samples_split=3,
                                         min_weight_fraction_leaf=0.0, n_estimators=140 * 2 + 1, n_jobs=-1,
                                         oob_score=False, verbose=0, warm_start=False)
            clf.fit(x_train, y_train)
            y_pred_rf = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred_rf)
            acc_list.append(acc)

            print('Overall accuracy of the model: ', accuracy_score(y_test, y_pred_rf))  # 整体精度
            print('Kappa coefficient of the model: ', cohen_kappa_score(y_test, y_pred_rf))  # Kappa系数

        # 画图
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure(facecolor='w')
        plt.plot(depth, acc_list, 'ro-', lw=1)
        plt.xlabel('随机森林决策树数量', fontsize=15)
        plt.ylabel('预测精度', fontsize=15)
        plt.title('随机森林决策树数量和过拟合', fontsize=18)
        plt.grid(True)
        plt.savefig(path, dpi=300)
        # plt.show()
        y_pred_rf = clf.predict(x_test)

        print('Overall accuracy of the model: ', accuracy_score(y_test, y_pred_rf))  # 整体精度
        print('Kappa coefficient of the model: ', cohen_kappa_score(y_test, y_pred_rf))  # Kappa系数

        matrix = confusion_matrix(y_test, y_pred_rf)
        kappa = cohen_kappa_score(y_test, y_pred_rf)
        dd = classification_report(y_test, y_pred_rf)
        acc_1 = accuracy_score(y_test, y_pred_rf)
        """
        # 特征重要性评定
        rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
        rnd_clf.fit(x, y)
        for name, score in zip(x, rnd_clf.feature_importances_):
            print(name, score)
        """
        return clf, matrix, dd, kappa, acc_1

    def save_model(self, clf, src):
        """
        保存模型到某处
        clf 模型
        src 路径
        """
        joblib.dump(clf, src)

    def get_model_predit(self, data, src):
        """
        调用模型实现预测
        输入原始数据
        src 模型路径
        返回预测值
        """
        getsavemodel = joblib.load(src)
        predity = getsavemodel.predict(pd.DataFrame(data))
        return predity


def get_data(file_path):
    datas = pd.read_csv(file_path, skiprows=1, names=['Structure', 'value'])
    datas = datas.values.tolist()

    smiles = []
    values = []

    for item in datas:
        sml = item[0]
        mol = Chem.MolFromSmiles(sml)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        smiles.append(arr)
        values.append(float(item[1]))
    return smiles, values


def split_train_test(data, ratio):
    # 划分训练集和测试集8:2
    num_total = data.shape[0]
    colum = data.shape[1]
    num_train = int(num_total * ratio)

    # 训练集
    x_train = data[0:num_train, :colum-1]
    y_train = data[0:num_train, colum-1:]

    # 测试集
    x_test = data[num_train:, :colum-1]
    y_test = data[num_train:, colum-1:]
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    positive_file = 'D:/Prediction-drug-molecular-activity/dataset/positive_after.csv'
    negative_file = 'D:/Prediction-drug-molecular-activity/dataset/negative_after.csv'
    pos_sml, pos_value = get_data(positive_file)
    neg_sml, neg_value = get_data(negative_file)

    features = pos_sml + neg_sml
    labels = pos_value + neg_value

    features = np.asarray(features)
    labels = np.asarray(labels).reshape([-1, 1])

    print(features.shape, labels.shape)
    data = np.c_[features, labels]
    print("data's shape is: ", data.shape)
    np.random.shuffle(data)

    x_train, y_train, x_test, y_test = split_train_test(data, 0.8)

    path = './depth'
    forest = Ctrain_forest()
    clf, matrix, dd, kappa, acc_1 = forest.train_forest(x_train, y_train, x_test, y_test, path)
    forest.plot_confusion_matrix(matrix, ['non-activate', 'activate'])
    print(dd)

