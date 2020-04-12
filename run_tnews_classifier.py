import pandas as pd
import codecs
import numpy as np
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


# 读取数据，划分训练集和验证集
df = pd.read_csv('data/tnews/toutiao_news_dataset.txt', delimiter = "_!_", names=['labels','text'], header = None, encoding='utf-8')
df = shuffle(df)   #shuffle数据
#把类别转换为数字，一共15个类别"民生故事","文化","娱乐","体育","财经","房产","汽车","教育","科技","军事","旅游","国际","证券股票","农业","电竞游戏"
class_le = LabelEncoder()
df.iloc[:,0] = class_le.fit_transform(df.iloc[:,0].values)


data_list = []
for data in df.iloc[:].itertuples():
    data_list.append((data.text, data.labels))

#取一部分数据做训练和验证
train_data  = data_list[0:10000]
valid_data = data_list[10000:11000]

maxlen = 100  # 设置序列长度为100，要保证序列长度不超过512

# 预训练模型
config_path = 'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

# 将词表中的词编号转换为字典
token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


# 重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')  # 不在列表的字符用[UNK]表示
        return R


tokenizer = OurTokenizer(token_dict)


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=32, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


# bert模型设置
bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))

x = bert_model([x1_in, x2_in])
x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
p = Dense(15, activation='softmax')(x)

model = Model([x1_in, x2_in], p)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(1e-5),
              metrics=['accuracy'])
model.summary()

train_D = data_generator(train_data)
valid_D = data_generator(valid_data)

model.fit_generator(
    train_D.__iter__(),
    steps_per_epoch=len(train_D),
    epochs=5,
    validation_data=valid_D.__iter__(),
    validation_steps=len(valid_D)
)
