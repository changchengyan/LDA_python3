# !/usr/bin/python
# -*- coding:utf-8 -*-

import time
import  os

import numpy as np
import pandas as pd
from gensim import corpora, models



## 获取当前根目录路径
dirPath = os.path.dirname(os.path.abspath(__file__))

## 当根目录的 生成的csv文件
csvName = 'comments'

## 匹配的词云 TXT 文件
wordCloud = 'LDA'

# 训练模型
num_topics = 30
# 全部语料训练的次数 数值越大 越耗时
passes = 2
#  大体理解是 控制运算速度
interations = 6000
# 进程数
workers = 3


def load_stopword():
    '''
    加载停用词表
    :return: 返回停用词的列表
    '''

    flag = validFilePath('LDA','txt')
    if not flag:
        return

    f_stop = open(dirPath+'/'+wordCloud+'.txt')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

def validFilePath(name,unit='csv'):
    if not os.path.exists(dirPath+'/'+name+'.'+unit):
        print('请在'+dirPath+'+目录下，新建一个 '+name+' 文件')
        return False
    return True



def exec(t_start,stop_words):


    if not validFilePath(csvName):
        return
    if not os.path.exists(dirPath+'/latitude'):
        print('当前根目录不存在 \" latitude \"  文件夹,请新建一个')
        return
    if os.listdir(dirPath+'/latitude'):
        print('当前根目录下 \" latitude \"  文件夹，有文件，请给于删除!')
        return
    print('')
    print('2.开始读入语料数据 ------ ')
    # 语料库分词并去停用词
    t_start = time.time()
    values= pd.read_csv(dirPath+'/'+csvName+'.csv', usecols=[csvName])
    ## 转换成 数组
    list = np.array(values)
    texts = [[word for word in str(item).strip().lower().split() if word not in stop_words] for item in list]

    print('读入语料数据完成，用时%.3f秒' % (time.time() - t_start))

    M = len(texts)
    print('文本数目：%d个' % M)

    print('')
    print('3.正在建立词典 ------')
    # 建立字典
    dictionary = corpora.Dictionary(texts)
    V = len(dictionary)
    print('字典大小为:'+ str(V))

    print('')
    print('正在生成字典文件')
    t_start = time.time()
    # 生成字典文件
    dictionary.save(dirPath+'/latitude/dictionary.dictionary')
    print('字典文件创建完成，用时%.3f秒' % (time.time() - t_start))

    print('')
    print('4.正在计算文本向量 ------')
    # 转换文本数据为索引，并计数
    t_start = time.time()
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('文本向量计算完成，用时%.3f秒' % (time.time() - t_start))
    ## 持久化分词
    print('')
    print('正在生成mm文件')
    t_start = time.time()
    corpora.MmCorpus.serialize(dirPath+'/latitude/corpus.mm',corpus)
    print('mm文件创建完成，用时%.3f秒' % (time.time() - t_start))

    print('5.正在计算文档TF-IDF ------')
    t_start = time.time()
    # 计算tf-idf值
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    print('建立文档TF-IDF完成，用时%.3f秒' % (time.time() - t_start))
    # 持久化 model模型 开始训练
    print('')
    print('正在进行模型训练和保存操作')
    t_start = time.time()
    lda_multi = models.ldamulticore.LdaMulticore(corpus=corpus_tfidf,id2word=dictionary,num_topics=num_topics,iterations=interations,workers=workers,batch=True,passes=passes)
    # 保存模型
    lda_multi.save(dirPath+'/latitude/lda.model')
    print('model训练并保存完成，用时%.3f秒' % (time.time() - t_start))

    print('')
    print('6.LDA模型拟合推断 ------')

    t_start = time.time()
    lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
                          alpha=0.01, eta=0.01, minimum_probability=0.001,
                          update_every=1, chunksize=100, passes=1)
    print('LDA模型完成，训练时间为\t%.3f秒' % (time.time() - t_start))

    # 随机打印某10个文档的主题
    num_show_topic = 10  # 每个文档显示前几个主题
    print('7.结果：10个文档的主题分布：--')
    doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
    idx = np.arange(M)
    np.random.shuffle(idx)
    idx = idx[:10]
    for i in idx:
        topic = np.array(doc_topics[i])
        topic_distribute = np.array(topic[:, 1])
        # print topic_distribute
        topic_idx = topic_distribute.argsort()[:-num_show_topic - 1:-1]
        print('第%d个文档的前%d个主题：' % (i, num_show_topic)), topic_idx
        print(topic_distribute[topic_idx])

    num_show_term = 7  # 每个主题显示几个词
    print('8.结果：每个主题的词分布：--')
    for topic_id in range(num_topics):
        print('主题#%d：\t' % topic_id)
        term_distribute_all = lda.get_topic_terms(topicid=topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:, 0].astype(np.int)
        print('词：\t', )
        for t in term_id:
            print(dictionary.id2token[t], )
        print('\n概率：\t', term_distribute[:, 1])




if __name__ == '__main__':

    print('1.初始化规避词列表 ------')
    # 开始的时间
    t_start = time.time()
    # 加载停用词表
    stop_words = load_stopword()
    print('读入规避词数据完成，用时%.3f秒' % (time.time() - t_start))

    exec(t_start,stop_words)



