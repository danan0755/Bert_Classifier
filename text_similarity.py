# -*- coding:utf-8 -*-

import jieba
from gensim import corpora
from gensim import models
from gensim import similarities

"""
    模型算法计算类-tfidf,lsi文本相似度计算
"""


class textSimMethod(object):
    """
        主要是计算文本相似度的算法
    """

    def __init__(self):
        pass

    def load_pre_data(self, **argvs):
        pass

    @staticmethod
    def get_sim_fun(standardAnswerText, answerText, modelType):
        # pass
        standardAnswer = standardAnswerText.strip().split("|")
        answer = answerText
        #
        ### standardAnswer分词
        all_doc_list = []
        for doc in standardAnswer:
            if len(doc)>0:
                doc_list = [word for word in jieba.cut(doc)]
                all_doc_list.append(doc_list)
        #
        ### answer分词
        doc_list_answer = [word for word in jieba.cut(answer)]

        ### 制作语料库 |制作词袋dd
        dictionary = corpora.Dictionary(all_doc_list)
        corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]

        # 将需要寻找相似度的（答案）分词列表，做成语料库 doc_answer_vec
        doc_answer_vec = dictionary.doc2bow(doc_list_answer)
        #
        ### 模型的训练
        if modelType == '1':
            # 将corpus语料库(初识语料库) 使用LsiModel模型进行训练
            modelTrain = models.LsiModel(corpus)
            modelName = 'LsiModel'
        elif modelType == '2':
            # 使用TfidfModel模型进行训练
            modelTrain = models.TfidfModel(corpus)
            modelName = 'TfidfModel'
        else:
            # 使用TfidfModel模型进行训练
            modelTrain = models.TfidfModel(corpus)
            modelName = 'TfidfModel'

        ### 文本相似度
        # 稀疏矩阵相似度，将主语料库corpus的训练结果作为初始值
        index = similarities.SparseMatrixSimilarity(modelTrain[corpus], num_features=len(dictionary.keys()))

        # 将语料库doc_answer_vec在语料库corpus的训练结果中的向量表示与语料库corpus的向量表示，做矩阵相似度计算
        sim = index[modelTrain[doc_answer_vec]]

        # 对下标和相似度结果进行一个排序，拿出相似度最高的结果
        # resultSort = sorted(enumerate(sim), key=lambda item: item[1],reverse=True)
        resultSort = sorted(enumerate(sim), key=lambda item: -item[1])

        ### 相似度结果格式转换
        result = []
        for simTuple in resultSort:
            rList = [simTuple[0] + 1, simTuple[1]]
            result.append(rList)

        return result

        # dict = {}
        # for i in range(len(sim)):
        #     dict[standardAnswer[i]] = sim[i]
        # return dict






###############
standardAnswerText = "你的名字是什么|你今年几岁了|你几岁了啊|你几岁哦"
answerText = ['你今年多大了','嘻嘻哈哈']
modelType = '2'
###############
for an in answerText:
    resultTest = textSimMethod.get_sim_fun(standardAnswerText, an, modelType)
    print(resultTest)


standardAnswerText = "这套带的学校是福民小学,深圳市福田区皇岗小学,水围小学,深圳市福田区皇岗中学|您好在的，很高兴为您服务，请问怎么称呼呢？"
answerText = "您好，请问怎么称呼您？"
modelType = '2'

resultTest = textSimMethod.get_sim_fun(standardAnswerText, answerText, modelType)
print(resultTest)




s1 = ['您好，请问怎么称呼您？','您好，这套房子还在的','现在是租客在住','这个是个人产权的','房产证照片我这边没有','您这边方便留个联系方式吗？']
s2 = "您好在的，很高兴为您服务，请问怎么称呼呢？"
for s in s1:
    ssim=textSimMethod.get_sim_fun(s,s2,'1')
    print(s,s2,ssim)


