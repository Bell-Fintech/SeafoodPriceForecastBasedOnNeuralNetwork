import os
import numpy as np
import pandas as pd
import csv
import sys

FEATURE_NUM = 5 # 特征值个数
TAG_NUM = 1 # 输出结果个数

def init_data(dir):
    data = pd.read_csv(open(dir,'r'),encoding = 'utf-8')
    class DataSets:
        def __init__(self):
            self.data = np.array(data)
            self.shape = self.data.shape
            #self.feature = sum(self.data[:,0:FEATURE_NUM].tolist(),[])
            self.feature = self.data[:,0:FEATURE_NUM]
            self.tag = self.data[:,FEATURE_NUM:FEATURE_NUM+TAG_NUM]
            #self.tag = sum(self.data[:,FEATURE_NUM:FEATURE_NUM+TAG_NUM].tolist(),[])
            self.feature_shape = self.data[:,0:FEATURE_NUM].shape
            self.tag_shape = self.data[:,FEATURE_NUM:FEATURE_NUM+TAG_NUM].shape
        def cat(self,num):
            print("----------datas-----------\n")
            print(self.data[0:num])
            print("----------features-----------\n")
            print(self.feature[0:num])
            print("----------tags-----------\n")
            print(self.tag[0:num])
        def info(self):
            print("the data info:\n")
            print("dir:\t",dir)
            print("data shape:\t",self.shape)
            print("feature nums:\t",self.feature_shape[1])
            print("tag nums:\t",self.tag_shape[1])
    Data = DataSets()
    return Data