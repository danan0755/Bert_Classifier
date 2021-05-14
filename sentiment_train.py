#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import jieba
import multiprocessing

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split

np.random.seed(2021)
import sys

sys.setrecursionlimit(1000000)
import yaml

'''情感分析训练接口'''

# 参数设置:
cpu_count = multiprocessing.cpu_count()  # 4
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10  # 所有频数超过10的词语
window_size = 7
n_epoch = 5
input_length = 100
maxlen = 100

batch_size = 32


class SentimentTrain():

    def load_data(self, neg_path=None, pos_path=None, neutral_path=None):
        if not neg_path:
            neg_path = '../data/neg.csv'
        if not pos_path:
            pos_path = '../data/pos.csv'
        if not neutral_path:
            neutral_path = '../data/neutral.csv'
        neg = pd.read_csv(neg_path, header=None, index_col=None)
        pos = pd.read_csv(pos_path, header=None, index_col=None, error_bad_lines=False)
        neu = pd.read_csv(neutral_path, header=None, index_col=None)
        combined = np.concatenate((pos[0], neu[0], neg[0]))
        y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int),
                            -1 * np.ones(len(neg), dtype=int)))

        return combined, y

    # 对句子经行分词，并去掉换行符
    def tokenizer(self, text):
        text = [jieba.lcut(document.replace('\n', '')) for document in text]
        return text

    def create_dictionaries(self, model=None, combined=None):
        if (combined is not None) and (model is not None):
            gensim_dict = Dictionary()
            gensim_dict.doc2bow(model.wv.vocab.keys(),
                                allow_update=True)
            #  freqxiao10->0 所以k+1
            w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引,(k->v)=>(v->k)
            w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量, (word->model(word))

            def parse_dataset(combined):  # 闭包-->临时使用
                data = []
                for sentence in combined:
                    new_txt = []
                    for word in sentence:
                        try:
                            new_txt.append(w2indx[word])
                        except:
                            new_txt.append(0)  # freqxiao10->0
                    data.append(new_txt)
                return data  # word=>index

            combined = parse_dataset(combined)
            combined = sequence.pad_sequences(combined, maxlen=maxlen)  # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
            return w2indx, w2vec, combined
        else:
            print('无训练数据')

    # 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
    def word2vec_train(self, combined):

        model = Word2Vec(size=vocab_dim,
                         min_count=n_exposures,
                         window=window_size,
                         workers=cpu_count,
                         iter=n_iterations)
        model.build_vocab(combined)
        model.train(combined, total_examples=model.corpus_count, epochs=model.iter)
        model.save('../model/Word2vec_model.pkl')
        index_dict, word_vectors, combined = self.create_dictionaries(model=model, combined=combined)
        return index_dict, word_vectors, combined

    def get_data(self, index_dict, word_vectors, combined, y):

        n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
        embedding_weights = np.zeros((n_symbols, vocab_dim))  # 初始化 索引为0的词语，词向量全为0
        for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
            embedding_weights[index, :] = word_vectors[word]
        x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
        y_train = keras.utils.to_categorical(y_train, num_classes=3)
        y_test = keras.utils.to_categorical(y_test, num_classes=3)
        return n_symbols, embedding_weights, x_train, y_train, x_test, y_test

    ##定义网络结构
    def train_lstm(self, n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
        model = Sequential()  #
        model.add(Embedding(output_dim=vocab_dim,
                            input_dim=n_symbols,
                            mask_zero=True,
                            weights=[embedding_weights],
                            input_length=input_length))
        model.add(LSTM(output_dim=50, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))  # Dense=>全连接层,输出维度=3
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        print("训练模型...")
        model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1)

        print("评估模型...")
        score = model.evaluate(x_test, y_test,
                               batch_size=batch_size)

        yaml_string = model.to_yaml()
        with open('../model/lstm.yml', 'w') as outfile:
            outfile.write(yaml.dump(yaml_string, default_flow_style=True))
        model.save_weights('../model/lstm.h5')
        print('test score:', score)
        return 'ok'

    def train(self, neg_path=None, pos_path=None, neutral_path=None):
        combined, y = self.load_data(neg_path, pos_path, neutral_path)
        combined = self.tokenizer(combined)
        index_dict, word_vectors, combined = self.word2vec_train(combined)
        n_symbols, embedding_weights, x_train, y_train, x_test, y_test = self.get_data(index_dict, word_vectors,
                                                                                           combined, y)
        result = self.train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)
        if result == 'ok':
            return '1'
        else:
            return '0'

if __name__ == '__main__':
    sentrain = SentimentTrain()
    r = sentrain.train()
    print(r)
