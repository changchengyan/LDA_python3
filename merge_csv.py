# !/usr/bin/python
# -*- coding:utf-8 -*-

import os
from glob  import glob

import pandas as pd


## 列的 label
columName = 'comments'

## 默认合并后 csv文件的名字
mergeName = 'comments'


## 获取当前根目录路径
dirPath = os.path.dirname(os.path.abspath(__file__))

## 获取当前 根目录下是否 含有csv 文件夹目录
dir_csv_existsFlag =  os.path.exists(dirPath+'/csv')


## 获取文件夹下的文件

def getFileNames():
    for root,dirs,files in os.walk(dirPath+'/csv'):
       return  files

## 合并csv文件
def mergeCsv():
    if (not dir_csv_existsFlag):
        print('您当前的目录下未含有 \"csv\目录!"')
        return

    mergeFilePath = dirPath+'/'+mergeName+'.csv'

    if not os.path.exists(mergeFilePath):
        f = open(mergeFilePath,'w')
        f.close()
    else:
        print('该文件已存在，请删除后完成生成!')
        return

    fileNames = getFileNames()

    if(len(fileNames)>0):
        combined_df = pd.concat(
            [
                pd.read_csv(csv_file, usecols=[columName],header=0,index_col=False,verbose=True)
                for csv_file in glob(dirPath+'/csv/*.csv')
            ]
        )
        combined_df.to_csv(mergeName+'.csv')

    else:
        print('请准备csv文件进行合并')
        return

if __name__ =='__main__':

    mergeCsv()











