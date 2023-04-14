import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from evaluation import ACC, F1_Score, Precision, Recall, MCC, Confusion_Matrix, plot_confusion_matrix
from wirteLog import make_print_to_file
import warnings
warnings.filterwarnings("ignore")

def get_data(file_path):
    datas = pd.read_csv(file_path, skiprows=1, names=['smiles','measured'])
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

    x_train = data[0:num_train, :colum-1]
    y_train = data[0:num_train, colum-1:]

    x_test = data[num_train:, :colum-1]
    y_test = data[num_train:, colum-1:]
    return x_train, y_train, x_test, y_test


def train(modelName, model, x_train, y_train, x_test, y_test):
    # cv_score = cross_val_score(model, x_train, y_train, cv=n_folds, scoring='recall')
    y_pred = model.fit(x_train, y_train).predict(x_test)

    pre = classification_report(y_test, y_pred, target_names=['non-activate', 'activate'], digits=5)
    print(pre)

    cm = Confusion_Matrix(y_test, y_pred)
    plot_confusion_matrix(cm, ['non-activate', 'activate'], path=modelName)

    metrics_name = [ACC, Precision, Recall, F1_Score, cohen_kappa_score]

    metrics_list = []
    for metric in metrics_name:
        score = metric(y_test, y_pred)
        metrics_list.append(score)

    print(100 * '-')
    print("%15s %15s %15s %15s %15s %15s" % ('modelName', 'ACC', 'Precision', 'Recall', 'f1_score', 'Kappa'))
    print("%15s %15.4f %15.4f %15.4f %15.4f %15.4f" % (modelName, metrics_list[0], metrics_list[1],
                                                  metrics_list[2], metrics_list[3],
                                                  metrics_list[4]))
    print(100 * '-')


# --------------------------------------------------------------------------------------------
if __name__ == '__main__':
    make_print_to_file(path='./')
    data_file = 'D:/SCHNet/dataset/data5_1-2.csv'

    features, labels = get_data(data_file)


    features = np.asarray(features)
    labels = np.asarray(labels).reshape([-1, 1])

    print(features.shape, labels.shape)
    data = np.c_[features, labels]
    print("data's shape is: ", data.shape)
    np.random.shuffle(data)
    x_train, y_train, x_test, y_test = split_train_test(data, 0.8)

    mnb = MultinomialNB()
    knn = KNeighborsClassifier()
    lr = LogisticRegression(penalty='l2')
    rf = RandomForestClassifier(n_estimators=8)
    dt = DecisionTreeClassifier(criterion="entropy")
    gb = GradientBoostingClassifier(n_estimators=200)
    svc = SVC(kernel='rbf', probability=True)

    modelNames = ['MultinomialNB', 'KNeighbors', 'Logistic', 'MultinomialNB', 'Logistic', 'RandomForest', 'SVC']
    modelDic = [mnb, knn, lr, rf, dt, gb, svc]
    for idx, model in enumerate(modelDic):
        train(modelNames[idx], model, x_train, y_train, x_test, y_test)





# plt.figure()
# plt.plot(np.arange(x_train.shape[0]), y_train, color='k', label='true y')
# color_list = ['r', 'b', 'g', 'y', 'c', 'w', 'k']
# linestyle_list = ['-', '.', 'o', 'v', '*', '-.', '+']
# for i, pre_y in enumerate(pre_y_list):
#     plt.plot(np.arange(x_train.shape[0]), pre_y_list[idx], color_list[idx], label=modelNames[idx])
# plt.title('classifiers train-train-result comparison')
# plt.legend(loc='upper right')
# plt.ylabel('real and predicted value')
# plt.show()
#
#
#


#
# # 模型应用
# print('regression prediction')
# new_point_set = x_test
# for i, new_point in enumerate(new_point_set):
#     new_pre_y = dt.predict(new_point.reshape(1, 2048))
#     print('predict for new point %d is:  %.2f, true value is %f' % (i + 1, new_pre_y, y_test[i]))  # 打印输出每个数据点的预测信息
