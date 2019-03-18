#!/usr/bin/env python
# coding=UTF-8
'''
@Description: The product price natural network 's test file
@version: 0.0.1
@Company: Student
@Author: StdKe
@LastEditors: StdKe
@Date: 2019-03-15 14:23:55
@LastEditTime: 2019-03-16 21:14:14
'''

import time
import tensorflow as tf 
import forward
import backward
import input_data

TEST_INTERVAL_SECS = 5

def test(sea_shrimp):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32,[None,forward.OUTPUT_NODE])
        y = forward.forwoard(x,None)
        data = input_data.read_data_sets("./data/test.csv",one_hot=True)
        X = data.train()[:,:forward.INPUT_NODE]
        Y = data.train()[:,forward.INPUT_NODE:forward.INPUT_NODE+1]
        
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVARAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy,feed_dict = {x:X,y_:Y})
                    print("After %s training steps ,test accuracy = %g"%(global_step,accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    sea_shrimp = input_data.read_data_sets("./data/test.csv",one_hot = True)
    test(sea_shrimp)

if __name__ == "__main__":
    main()

