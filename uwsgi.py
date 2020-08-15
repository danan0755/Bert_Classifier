import numpy as np
import pandas as pd
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from flask import Flask, request, jsonify
from keras.layers import Lambda, Dense
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

app = Flask(__name__)

# 因为tensorflow是动态图，所以graph要作为全局变量，如果是局部变量，则global graph
global graph
graph = tf.get_default_graph()

# maxlen = 128
# batch_size = 32
# config_path = '/data/pymodel/project_lyj_model/albert_base_zh/albert_config.json'
# checkpoint_path = '/data/pymodel/project_lyj_model/albert_base_zh/model.ckpt-best'
# dict_path = '/data/pymodel/project_lyj_model/albert_base_zh/vocab_chinese.txt'
# model_path = '/data/pymodel/project_lyj_model/model/model_intent1/best_model.h5'
# train_data_path = '/data/pymodel/project_lyj_model/data/train.tsv'

# 读取配置文件
maxlen = int(read_ini('intent', 'maxlen'))
batch_size = int(read_ini('intent', 'batch_size'))
config_path = read_ini('intent', 'config_path')
checkpoint_path = read_ini('intent', 'checkpoint_path')
dict_path = read_ini('intent', 'dict_path')
model_path = read_ini('intent', 'model_path1')
train_data_path = read_ini('intent', 'train_data_path')



def get_labels():
    df = pd.read_csv(train_data_path, delimiter="\t", names=['labels', 'text'],
                     header=0, encoding='utf-8', engine='python')
    labels_df = df[['labels']]
    labels_df = labels_df.drop_duplicates(ignore_index=True)

    labels = []
    for data in labels_df.iloc[:].itertuples():
        labels.append(data.labels)
    return labels


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# labels数量
dense_units = len(get_labels())
print(dense_units)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=dense_units,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

global model
model = keras.models.Model(bert.model.input, output)
model.load_weights(model_path)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def load_data(filepath):
    df = pd.read_csv(filepath, delimiter="\t", names=['labels', 'text'], header=0,
                     encoding='utf-8', engine='python')
    # df = shuffle(df)  # shuffle数据
    class_le = LabelEncoder()
    df.iloc[:, 0] = class_le.fit_transform(df.iloc[:, 0].values)

    lines = []
    for data in df.iloc[:].itertuples():
        lines.append((data.text, data.labels))
    return lines, class_le


# 加载数据集
train_data, class_le = load_data(train_data_path)


# 输入的是用户问句，输出是预测label和概率值
def predict(data):
    with graph.as_default():
        for x_true, y_true in data:
            y_pred = model.predict(x_true)
            score = np.max(y_pred, axis=1)[0]
            y_pred = y_pred.argmax(axis=1)
            y_pred = class_le.inverse_transform(y_pred)
            y_pred = y_pred[0]
            return y_pred, score



# 意图识别接口
@app.route('/intent', methods=["GET"])
def intent():
    try:
    	# 接收处理GET数据请求
        query = request.args.get('query')
        query_label = [(query_pun, 0)]
        test_generator = data_generator(query_label, batch_size)
        label, score = predict(test_generator)
        return label, score
    except Exception as e:
        print(e)
if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(debug=False)
