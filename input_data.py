#!/usr/bin/env python
# coding=UTF-8
'''
@Description: input the data of training stes
@version: 0.0.1
@Company: Student
@Author: StdKe
@LastEditors: StdKe
@Date: 2019-03-14 14:11:12
@LastEditTime: 2019-03-16 21:13:14
'''

import tensorflow as tf
import os
import numpy as np
import csv
import pandas as pd
import sys
import random

N = 3 # Fibonacci number's size
PROPORRION = 0.25 # the PROPORRION of test sets

'''
@name: Fibonacci_Yield_tool
@description: use these functons to produce Fibonacci list as the index of next_batch
@msg: let n = 3 ,Fibonacci list grows up too fast, so using smaller seed number
@param {int} {n}
@return: Fibonacci number list
'''
def Fibonacci_Yield_tool(n):
  a, b = 0, 1
  while n > 0:
    yield b
    a, b = b, a + b
    n -= 1
def Fibonacci_Yield(n):
    return list(Fibonacci_Yield_tool(n))

'''
@name: read_data_sets
@description: the functon to read data of train sets,
              when call this functon,you will get an object 
              of train sets which has some common sets' functons 
@msg: obj.info => train data's info of matix
      obj.size => train data's shape
      obj.current_step => the current step when training the network,
                          which can save the massage of current sets
      obj.index_list => use the Fibonacci number to random-index the every batch
@param {dir} {data_dir} {one_hot_bool} {True}
@return: {object} {DataSets}

'''
def read_data_sets(data_dir,one_hot = True):
  data = pd.read_csv(open(data_dir,'rU'),encoding = 'utf-8')
  class DataSets:
    def __init__(self):
      self.info = data[:].values
      self.size = self.info.shape
      self.current_step = 0
      self.index_list = Fibonacci_Yield(N)
    '''
    @name: next_batch
    @description: call this functon to get the next batch to feet the network ,if index out of the range,use random-index move in cycles
    @msg: 
    @param {self} {int} {BATCH_SIZE}
    @return: the index list of x,y [x,y]
    '''
    def next_batch(self,BATCH_SIZE):
      self.current = self.current_step
      if(self.current_step >= self.size[0]): self.current_step = random.sample(self.index_list,1)[0]
      self.current_step = self.current_step+BATCH_SIZE
      return [self.info[self.current:self.current_step][:,0:self.size[1]-1],self.info[self.current:self.current_step][:,self.size[1]-1:self.size[1]] ]
    '''
    @name: test_list
    @description: call this functon to get the test list of train sets , size  is total's 25% 
    @msg: 
    @param {self}
    @return: test_list
    '''
    def test_list(self):
      return self.info[:,0:self.size[1]-1][int(self.size[0]-self.size[0]*PROPORRION):self.size[0],:]
    '''
    @name: test_price
    @description: call this functon to get the test price list of train sets , size  is total's 25% 
    @msg: 
    @param {self}
    @return: 
    '''
    def test_price(self):
      return self.info[:,self.size[1]-1:self.size[1]][int(self.size[0]-self.size[0]*PROPORRION):self.size[0],:]
    def train(self):
      return self.info[:][int(self.size[0]-self.size[0]*(1-PROPORRION)):self.size[0],:]

    
  DataSets = DataSets()
  return DataSets
