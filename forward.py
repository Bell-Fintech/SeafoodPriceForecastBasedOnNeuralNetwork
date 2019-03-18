#!/usr/bin/env python
# coding=UTF-8
'''
@Description: the Sea shrimp price forecast's neural networks' forwoard 
@version: 0.0.1
@Company: Student
@Author: StdKe
@LastEditors: StdKe
@Date: 2019-03-15 09:20:42
@LastEditTime: 2019-03-16 18:42:57
'''

import tensorflow as tf

INPUT_NODE = 5 # five features
OUTPUT_NODE = 1 # output is the price
LAYER1_NODE = 5 # first layer nodes
LAYER2_NODE = 5 # second layer nodes

'''
@name: get_weight
@description: generateing the weight "w"
@msg: The parameter satisfies the truncated normal distribution and use the regularizer
@param {node_number}{shape},{tf_regularizer}{regularizer}
@return: w
'''
def get_weight(shape,regularizer):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

'''
@name: get_bias
@description: generate bias term
@msg: 
@param: {node_number}{bias}
@return: b
'''
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

'''
@name: forwoard
@description: Forward propagation of neural networks ，Contains two hidden layers
@msg: Except for the last layer of the output layer, all other layer outputs are passed the activation function rule
@param {matrix} {x} ,{tf_regularizer} {regularizer}
@return: y
'''
def forwoard(x,regularizer):
    w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

    w2 = get_weight([LAYER1_NODE,LAYER2_NODE],regularizer)
    b2 = get_bias([LAYER2_NODE])
    y2 = tf.nn.relu(tf.matmul(y1,w2) + b2)
    
    w3 = get_weight([LAYER2_NODE,OUTPUT_NODE],regularizer)
    b3 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y2,w3) + b3

    return y
    
    