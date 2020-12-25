#!/usr/bin/env python
# -*- coding: utf-8 -*-

# author ChenYongSheng
# date 20201222

import fasttext
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from intent.intent_utils import get_label_dict

'''模型训练'''

trainDataFile = 'data/8qi/train.txt'

model = fasttext.train_supervised(
    input=trainDataFile,
    dim=200,
    epoch=50,
    lr=0.1,
    lrUpdateRate=50,
    minCount=3,
    loss='softmax',
    wordNgrams=2,
    bucket=1000000)

# model.save_model("model/fasttext_model.bin")

testDataFile = 'data/8qi/test.txt'

# model = fasttext.load_model('model/fasttext_model.bin')

result = model.test(testDataFile)
print('测试集上数据量', result[0])
print('测试集上准确率', result[1])
print('测试集上召回率', result[2])


# print(model.labels)
# print(model.words)

# 计算分类的metrics
# 绘制precision、recall、f1-score、support报告表
def eval_model(y_true, y_pred, labels):
    # 计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    # 计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision': p,
        u'Recall': r,
        u'F1': f1,
        u'Support': s
    })
    res2 = pd.DataFrame({
        u'Label': ['总体'],
        u'Precision': [tot_p],
        u'Recall': [tot_r],
        u'F1': [tot_f1],
        u'Support': [tot_s]
    })
    res2.index = [99999]
    res = pd.concat([res1, res2])
    return res[['Label', 'Precision', 'Recall', 'F1', 'Support']]


label_dict_file = 'data/8qi/label_dict.xls'
cate_dic = get_label_dict(label_dict_file)
dict_cate = dict(('__label__{}'.format(k), v) for k, v in cate_dic.items())
y_true = []
y_pred = []
with open('data/8qi/test.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        splits = line.split(" ")
        label = splits[0]
        words = [" ".join(splits[1:])]
        label = dict_cate[label]
        y_true.append(label)
        y_pred_results = model.predict(words)[0][0][0]
        y_pred.append(dict_cate[y_pred_results])
print("y_true = ", y_true[:5])
print("y_pred = ", y_pred[:5])
print('y_true length = ', len(y_true))
print('y_pred length = ', len(y_pred))

print('keys = ', list(cate_dic.keys()))

eval_model(y_true, y_pred, list(cate_dic.keys()))

import jieba

text = "这个房子安静吗"
words = [word for word in jieba.lcut(text)]
print('words = ', words)
data = " ".join(words)

# predict
results = model.predict([data])
y_pred = results[0][0][0]
print("y_pred results = ", str(y_pred).replace('__label__', ''), dict_cate[y_pred])
