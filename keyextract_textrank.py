#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 采用TextRank方法提取文本关键词
import pandas as pd
import jieba.analyse
import os
from download_db_data import DownloadData
import multiprocessing

"""
       TextRank权重：

            1、将待抽取关键词的文本进行分词、去停用词、筛选词性
            2、以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图
            3、计算图中节点的PageRank，注意是无向带权图
"""

# jieba.load_userdict("data/userdict.txt")
jieba.load_userdict("/opt/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/userdict.txt")

# jieba.analyse.set_stop_words("data/stopWord.txt")  # 加载自定义停用词表
jieba.analyse.set_stop_words(
    "/opt/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/stopWord.txt")  # 加载自定义停用词表

# other_words = [word.rstrip('\n') for word in open('data/other_word.txt', 'r', encoding='utf-8')]
other_words = [word.rstrip('\n') for word in
               open('/opt/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/other_word.txt',
                    'r', encoding='utf-8')]

# 处理标题和摘要，提取关键词
def getKeywords_textrank(data_file, topK, stopwords):
    df = pd.read_csv(data_file, names=['from_account', 'msg_times_tamp', 'body', 'odate', 'omonth'])
    from_account_list, msg_times_tamp_list, body_list, odate_list, omonth_list = \
        df['from_account'], df['msg_times_tamp'], df['body'], df['odate'], df['omonth']
    from_accounts, msg_times_tamps, bodys, odates, omonths, keys, other_keys = [], [], [], [], [], [], []
    for index in range(len(body_list)):
        # print("\"", body_list[index], "\"", " \Keywords - TextRank :")
        # keywords = jieba.analyse.textrank(str(body_list[index]), topK=topK,
        #                                   allowPOS=('n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd'))  # TextRank关键词提取，词性筛选
        keywords = jieba.analyse.textrank(str(body_list[index]), topK=topK)  # TextRank关键词提取，词性筛选
        cut_words = jieba.lcut(str(body_list[index]))
        if keywords:
            word_split = " ".join(keywords)
            # print(word_split)
            keys.append(word_split)
        else:
            # 切词
            words = [w for w in cut_words if w not in stopwords]
            word_split = " ".join(words)
            keys.append(word_split)
        # 其他词
        o_words = ''
        for word in cut_words:
            if word in other_words and len(word) > 1:
                o_words = word + ' ' + o_words
        other_keys.append(o_words)
        # print(o_words)
        from_accounts.append(from_account_list[index])
        msg_times_tamps.append(msg_times_tamp_list[index])
        bodys.append(body_list[index])
        odates.append(odate_list[index])
        omonths.append(omonth_list[index])

    result = pd.DataFrame({"from_account": from_accounts, "msg_times_tamp": msg_times_tamps,
                           "body": bodys, "odate": odates, "omonth": omonths, "key": keys, "other_key": other_keys},
                          columns=['from_account', 'msg_times_tamp', 'body', 'odate', 'omonth', 'key', 'other_key'])
    return result




def run():
    dl = DownloadData()
    im_odate_path = '/data/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/im_odate.csv'
    odates = dl.get_df(im_odate_path)
    stopwords = [w.strip() for w in
                 open('/opt/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/stopWord.txt', 'r',
                      encoding='utf-8').readlines()]
    for odate in odates:
        data_file = '/data/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/download/im用户咨询{}.csv'.format(
            odate)
        result = getKeywords_textrank(data_file, 5, stopwords)
        upload_file = '/data/pymodel/project_lyj_model/nlp_project/keyword_extraction/data/upload4/im用户咨询_keywords{}.csv'.format(
            odate)
        if not os.path.exists(upload_file):
            result.to_csv(upload_file, encoding='utf-8', index=False, header=0)
    print('-------finish-------')


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=30)
    pool.apply_async(run)
    pool.close()
    pool.join()

    # data_file = 'data/im用户咨询2020-11-13.csv'
    # stopwords = [w.strip() for w in open('data/stopWord.txt', 'r', encoding='utf-8').readlines()]
    # result = getKeywords_textrank(data_file, 2, stopwords)
    # print(result)
    # result.to_csv('result/im_TextRank.csv', encoding='utf-8', index=False, header=0)
