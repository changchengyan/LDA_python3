#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
import logging
import time

from gensim import corpora, models
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s : ', level=logging.INFO)

## 获取当前根目录路径
dirPath = os.path.dirname(os.path.abspath(__file__))


def drawLine(topic,perplexity_list):
    x = topic
    y = perplexity_list
    plt.plot(x,y,color="red",linewidth=2)
    plt.xlabel("Number of Topic")
    plt.ylabel("Perplexity")
    plt.show()
    print('请看图^_^')


def perplexity(ldamodel, testset, dictionary, size_dictionary, num_topics):
    print('')
    print ('ldamodel 信息提示: \n')
    print ('测试集的数量: %s; 字典大小: %s; 主题数量: %s'%(len(testset), size_dictionary, num_topics))
    prep = 0.0 ## 初始化宽容度
    prob_doc_sum = 0.0 # 初始化文档数量
    topic_word_list = [] # 存储可能的主题词
    t_start = time.time()
    print('开始计算 宽容度')
    for topic_id in range(num_topics):
        topic_word = ldamodel.show_topic(topic_id, size_dictionary)
        dic = {}
        for word, probability in topic_word:
            dic[word] = probability
        topic_word_list.append(dic)
    doc_topics_ist = [] #构造列表元组
    for doc in testset:
        doc_topics_ist.append(ldamodel.get_document_topics(doc, minimum_probability=0))
    testset_word_num = 0

    # 宽容度列表 y轴 绘制图表用
    perplexity_list = []
    # 宽容度 x轴 绘制图表用
    topic =[]
    for i in range(len(testset)):
        topic.append(5 * (3 * i + 1))

    print(topic)


    for i in range(len(testset)):
        prob_doc = 0.0 # 初始化可能的文档 默认是第一个
        doc = testset[i]
        doc_word_num = 0 # 初始化 文档中 词汇的个数
        for word_id, num in doc :
            prob_word = 0.0 # 初始化词汇
            doc_word_num += num
            word = dictionary[word_id]
            for topic_id in range(num_topics):
                prob_topic = doc_topics_ist[i][topic_id][1]
                prob_topic_word = topic_word_list[topic_id][word]
                prob_word += prob_topic*prob_topic_word
            prob_doc += math.log(prob_word) # 求和 p(d) = sum(log(p(w)))
        prob_doc_sum += prob_doc
        testset_word_num += doc_word_num
        # 获取把各个文档中收集的词汇汇总，然后塞到列表中
        perplexity_list.append(math.exp(-prob_doc/doc_word_num))

    print(perplexity_list)

    prep = math.exp(-prob_doc_sum/testset_word_num) # 计算宽容度   perplexity = exp(-sum(p(d)/sum(Nd))
    print('完成宽容度计算，当前耗时\t%.3f秒' % (time.time() - t_start))
    print ("模型的宽容度 是 : %s"%prep)
    print('')
    # 绘制线图
    print('开始绘制线图')
    drawLine(topic,perplexity_list)

    return prep





if __name__ == '__main__':

    dir_flag = os.path.exists(dirPath+'/latitude')

    if dir_flag:
        middatafolder =dirPath+'/latitude/'

        dictionary_path = middatafolder + 'dictionary.dictionary'
        corpus_path = middatafolder + 'corpus.mm'
        ldamodel_path = middatafolder + 'lda.model'
        dictionary = corpora.Dictionary.load(dictionary_path)
        corpus = corpora.MmCorpus(corpus_path)
        lda_multi = models.ldamodel.LdaModel.load(ldamodel_path)
        # 主题数
        num_topics = 10
        # 测试集
        testset = []
        # 样本数量率
        sampleNum = 300
        # sample 1/300
        # 当前测试 样本集 过少 就 算作1

        for i in range ( int(corpus.num_docs/sampleNum)):
            testset.append(corpus[i*sampleNum])
        prep = perplexity(lda_multi, testset, dictionary, len(dictionary.keys()), num_topics)
    else:
        print('您的模型库还未生成!')