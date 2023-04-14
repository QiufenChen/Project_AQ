# -*- coding:utf-8 -*-
"""Sample training code
"""
import numpy as np
import pandas as pd
import argparse
import torch as th
import torch.nn as nn
from sch import SchNetModel
from mgcn import MGCNModel
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher
from evaluation import ACC, F1_Score, Precision, Recall, MCC, AUC
import warnings
warnings.filterwarnings('ignore')


def train(epochs=80, dataset='', save=''):
    print("start")
    train_dir = "./"
    train_file = dataset+"data5_preprocess.csv"
    # print(train_file)
    taotal_data = TencentAlchemyDataset()
    taotal_data.mode = "Train"
    taotal_data.transform = None
    taotal_data.file_path = train_file
    taotal_data._load()
    # print(type(taotal_data))
    # 5292 = 0.8 * 5292 + 0.2 * 5292 = 4234 + 1058
    num = len(taotal_data)
    train_set, test_set = th.utils.data.random_split(taotal_data, [int(0.8*num), int(0.2*num)])
    print(len(train_set), len(test_set))

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=10,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=10,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=4,
    )

    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    model = SchNetModel(norm=False, output_dim=2)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=10, threshold=0.0000001, threshold_mode='rel', cooldown=0, min_lr=0.000001, eps=1e-08, verbose=False)

    model.train()
    for epoch in range(epochs):
        w_loss, w_acc, w_prec, w_recall, w_f1, w_auc = 0, 0, 0, 0, 0, 0
        model.train()

        for idx, batch in enumerate(train_loader):

            y_true = batch.label.squeeze(-1).long().to(device)
            y_pred = model(batch.graph)

            # print(y_true.shape, y_pred.shape)
            loss = loss_fn(y_pred, y_true)

            y_pred = th.argmax(y_pred, axis=1)
            
            acc = ACC(y_true, y_pred)
            prec = Precision(y_true, y_pred)
            recall = Recall(y_true, y_pred)
            f1 = F1_Score(y_true, y_pred)
            auc = AUC(y_true, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_loss += loss.detach().item()
            w_acc += acc
            w_prec += prec
            w_recall += recall
            w_f1 += f1
            w_auc += auc

        w_loss /= idx + 1
        w_acc /= idx + 1
        w_prec /= idx + 1
        w_recall /= idx + 1
        w_f1 /= idx + 1
        w_auc /= idx + 1
        scheduler.step(w_loss)

        print("Epoch {:2d}, train_loss: {:.4f}, train_acc: {:.4f}, "
              "train_prec:{:.4f}, train_recall: {:.4f}, train_f1: {:.4f}, train_auc: {:.4f}"
              .format(epoch, w_loss, w_acc, w_prec, w_recall, w_f1, w_auc))

        # output = 'Epoch: ' + str(epoch) + 'train_loss: ' + str(w_loss) + 'train_mae: ' + str(w_mae)
        # save_train.append(output)

        model.eval()
        val_loss, val_acc, val_prec, val_recall, val_f1, val_auc = 0, 0, 0, 0, 0, 0
        with th.no_grad():
            for jdx, batch in enumerate(test_loader):
              
                y_true = batch.label.squeeze(-1).long().to(device)

                y_pred = model(batch.graph)
                loss = loss_fn(y_pred, y_true)

                y_pred = th.argmax(y_pred, axis=1)

                acc = ACC(y_true, y_pred)
                prec = Precision(y_true, y_pred)
                recall = Recall(y_true, y_pred)
                f1 = F1_Score(y_true, y_pred)
                auc = AUC(y_true, y_pred)

                val_loss += loss.detach().item()
                val_acc += acc
                val_prec += prec
                val_recall += recall
                val_f1 += f1
                val_auc += auc

            val_loss /= jdx + 1
            val_acc /= jdx + 1
            val_prec /= jdx + 1
            val_recall /= jdx + 1
            val_f1 /= jdx + 1
            val_auc /= jdx + 1

            # output = 'Epoch: ' + str(epoch) + 'val_loss: ' + str(val_loss) + 'val_mae: ' + str(val_mae)
            # save_val.append(output)
            print("Epoch {:2d}, val_loss: {:.4f}, val_acc: {:.4f}, "
                  "val_prec:{:.4f}, val_recall: {:.4f}, val_f1: {:.4f}, val_auc: {:.4f}"
                  .format(epoch, val_loss, val_acc, val_prec, val_recall, val_f1, val_auc))

        if epoch % 10 == 0:
            th.save(model.state_dict(), save + "/sch_solub" + str(epoch) + '.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch, mgcn)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=10000)
    parser.add_argument("--dataset", help="dataset to train", default="")
    parser.add_argument(
        "--save", help="folder path to save results", default="")
    # device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    # device=th.device("cpu")

    args = parser.parse_args()
    args.epochs = 100
    args.dataset = './dataset/'
    args.save = './model/'
    train(int(args.epochs), args.dataset, args.save)
