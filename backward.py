#!/usr/bin/env python
# coding=UTF-8
'''
@Description: the Sea shrimp price forecast's neural networks' backwoard 
@version: 
@Company: Student
@Author: StdKe
@LastEditors: StdKe
@Date: 2019-03-15 12:47:49
@LastEditTime: 2019-03-16 20:49:00
'''
import tensorflow as tf
import forward
import input_data
import os

sea_shrimp = input_data.read_data_sets('./data/test.csv',one_hot= True)
BATCH_SIZE = 32 # The number of feeds per neural network
#LEARNING_RATE_BASE = 0.01 # The base of larning rate
#LEARNING_RATE_DECAY = 0.09 # the dacay of larning rate
REGULARIZER = 0.0001 # the parameter of regularizer 
STEPS = 50000 # the steps of train
MOVING_AVARAGE_DECAY = 0.99 # The decay of moving avarage 
MODEL_SAVE_PATH = "./model/" # The path of model saving 
MODEL_NAME = "sea_shrimp_model" #The name of model saving 
sea_shrimp_size = sea_shrimp.size[0]

'''
@name: backward
@description: Forward propagation of neural networks
@msg: loss function :
            customize loss function
            mse loss function
@param {matrix} {sea_shrimp}
@return: 
'''
def backward(sea_shrimp):
    x = tf.placeholder(tf.float32,[None,forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
    y = forward.forwoard(x,REGULARIZER)
    global_step = tf.Variable(0,trainable = False)
    #loss = tf.reduce_sum(tf.where(tf.greater(y,y_),COST(y-y_),PROFIT(y_-y))) + tf.add_n(tf.get_collection('losses'))
    #loss = tf.reduce_mean(tf.square(y_-y))+tf.add_n(tf.get_collection('losses'))
    loss = tf.reduce_mean(tf.square(y_-y))
    
    '''
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        sea_shrimp_size/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    '''

    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVARAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name = 'train')
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range (STEPS):
            xs,ys = sea_shrimp.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            if i % 1000 == 0:
                print("After %d trainging steps ,loss on traing batch is %g.."%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
    backward(sea_shrimp)

if __name__ == "__main__":
    main()

        
            
    
