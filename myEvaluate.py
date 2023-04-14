# encoding: utf-8
"""
@author: julse@qq.com
@time: 2020/4/15 20:47
@desc:
"""
import tensorflow as tf
import tensorflow.keras.backend as K



#精确率评价指标
def metric_precision(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP+ K.epsilon())
    return precision
#召回率评价指标
def metric_recall(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    recall=TP/(TP+FN+ K.epsilon())
    return recall
#F1-score评价指标
def metric_F1score(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    precision=TP/(TP+FP+ K.epsilon())
    recall=TP/(TP+FN+ K.epsilon())
    F1score=2*precision*recall/(precision+recall+ K.epsilon())
    return F1score
# MCC
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
    # return MCC(y_true, y_pred)

'''
old model need this
'''
def MCC(y_true,y_pred):
    TP = tf.reduce_sum(y_true * tf.round(y_pred))
    TN = tf.reduce_sum((1 - y_true) * (1 - tf.round(y_pred)))
    FP = tf.reduce_sum((1 - y_true) * tf.round(y_pred))
    FN = tf.reduce_sum(y_true * (1 - tf.round(y_pred)))
    myMCC = (TP*TN - FP*FN)*1.0/(tf.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))+K.epsilon())
    return myMCC
def metric_ACC(y_true,y_pred):
    TP=tf.reduce_sum(y_true*tf.round(y_pred))
    TN=tf.reduce_sum((1-y_true)*(1-tf.round(y_pred)))
    FP=tf.reduce_sum((1-y_true)*tf.round(y_pred))
    FN=tf.reduce_sum(y_true*(1-tf.round(y_pred)))
    acc=(TP+TN)/(TP+FP+TN+FN+K.epsilon())
    return acc
class MyEvaluate:
    metric = ['acc',metric_precision, metric_recall, metric_F1score, matthews_correlation]
    metric_json = {
        'acc':'acc',
        'metric_precision': metric_precision,
        'metric_recall': metric_recall,
        'metric_F1score': metric_F1score,
        'MCC': MCC,
        'matthews_correlation': matthews_correlation
    }
    def evaluate_manual(self,y_true,y_pred):
        # y_true = tf.constant(list(y_true), dtype=float)
        # y_pred = tf.constant(list(y_pred),dtype=float)
        y_true = tf.constant(list(y_true),dtype=float)
        y_pred = tf.constant(y_pred,dtype=float)
        return [K.eval(metric_ACC(y_true, y_pred)),
                K.eval(metric_precision(y_true, y_pred)),
                K.eval(metric_recall(y_true, y_pred)),
                K.eval(metric_F1score(y_true, y_pred)),
                K.eval(matthews_correlation(y_true, y_pred))]

if __name__ == '__main__':
    import pandas as pd
    import tensorflow as tf
    result_out = '/home/jjhnenu/data/PPI/release/result_in_paper/alter_ratio/p_fw_v1_train_validate_v2_fixpositive/1/validate/result.csv'
    df = pd.read_csv(result_out)
    y_true = df['real_label']
    # y_pred = df['predict']   # [0.55012226, 0.55494505, 0.49509802, 0.523316, 0.10058646]
    y_pred = df['predict_label']   #  [0.55012226, 0.55494505, 0.49509802, 0.523316, 0.10058646]

    # y_true = tf.constant(list(y_true),dtype=float)
    # y_pred = tf.constant(list(y_pred))
    # # print(mean(K.eval(K.binary_crossentropy(y_true, y_pred))))
    # print('acc\t\t\t',K.eval(metric_ACC(y_true, y_pred)))
    # print('metric_precision\t\t',K.eval(metric_precision(y_true, y_pred)))
    # print('metric_recall\t\t',K.eval(metric_recall(y_true, y_pred)))
    # print('metric_F1score\t\t',K.eval(metric_F1score(y_true, y_pred)))
    # print('matthews_correlation\t',K.eval(matthews_correlation(y_true, y_pred)))
    # print('MCC\t\t\t',K.eval(MCC(y_true, y_pred)))

    print('[acc,metric_precision, metric_recall, metric_F1score, matthews_correlation]\n',MyEvaluate().evaluate_manual(y_true,y_pred))

    # y_true = tf.constant([0.0, 1.0])
    # y_pred = tf.constant([0.8, 0.2])
    # K.eval(K.binary_crossentropy(y_true, y_pred))

