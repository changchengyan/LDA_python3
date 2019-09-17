
## 项目需求 
    
    用python写一个 基于lda模型的主题词提取 有15个csv文件，读取其中的comments那一列，整合到一起 ，然后用lda模型提取出主题词
    在利用宽容度算法 得出 所需数值并绘制 线性图



##文件目录


 comments.csv   // 生成的csv文件
│  LDA.txt      // 写入 需要规避的分词
│  merge_csv.py  // 合并csv文件
│  perplexity.py  // 宽容度 计算
│  README.md   // 就是我
│  wordCloud_lda.py
│
├─csv  // 一大堆 csv文件 
│      Austin_180514_reviews.csv
│      Boston_180517_reviews.csv
│
└─latitude //生成的LDA 模型数据
        corpus.mm
        corpus.mm.index
        dictionary.dictionary
        lda.model
        lda.model.expElogbeta.npy
        lda.model.id2word
        lda.model.state
        
        
 
## 执行流程 
    1、 先执行 merge_csv.py  生成想要的数据  （执行前、先删除 comments.csv 文件）
    2、 然后 执行 wordCloud_lda.py 用来完成运算 (执行前先删除 latitude下的所有文件)
    3、 在执行 perplexity.py 完善 宽容度计算 和绘制线状图
    

## 大量使用的 包 （重要的是在 python3环境下）

    1、pandas      V0.25.1
    2、numpy       V1.17.2
    3、gensim      V3.8.0
    4、matplotlib  V3.1.1
