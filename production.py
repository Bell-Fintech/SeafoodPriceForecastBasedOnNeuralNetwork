# -*- coding: utf-8 -*-

"""
regression  RNN LSTM
tensorboard
plt.plot
RNN
LSTM
"""
#tensorboard --logdir='logs'
# google -> http://0.0.0.0:6006


#分类使用[(batch_size, output_size)*steps] 中最后一个step的值;
#分类使用或者描述为(batch_size, n_step, output_size)中(batch_size, -1, output_size)


#回归问题中，尽管可能输入和输出维度是1，
#但是可以time_steps=20,即把20个点当成个序列，这时候就要考虑每一步的output，合起来就是20个输出，即一个序列。


#import packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import input_data
from sklearn import preprocessing
import matplotlib.pyplot as plt

tf.reset_default_graph()
#define hypeparameter
BATCH_START = 0 # 建议batch data 的index
TIME_STEPS = 20 # backpropagation through time 的 time_steps
BATCH_SIZE = 20
INPUT_SIZE = 5 #输入数据的size
OUTPUT_SIZE = 1 # 输出的size
CELL_SIZE = 10 # RNN 的 hidden unit size
LR = 0.02 # 学习率
BATCH_START_TEST = 0
# 加载
data = input_data.init_data('_test.csv')
x = data.feature
y = data.tag
data.info()
data.cat(10)
# 标准化数据
ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(x)
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(y.reshape(-1, 1))
print(ss_x)
print(ss_y)





def get_batch_data():
    global BATCH_START, TIME_STEPS,train_x,trian_y
    x_part1 = train_x[BATCH_START : BATCH_START+TIME_STEPS*BATCH_SIZE]
    y_part1 = train_y[BATCH_START : BATCH_START+TIME_STEPS*BATCH_SIZE]
    #print('time -- ', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)
    seq =x_part1.reshape((BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
    res =y_part1.reshape((BATCH_SIZE, TIME_STEPS ,1))
    BATCH_START += TIME_STEPS
    # returned seq, res and xs: shape (batch, step, input)
    #np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return[seq,res]

def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    print('xs.shape=',xs.shape)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    import matplotlib.pyplot as plt
    plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    plt.show()
    print('增加维度前:',seq.shape)
    print( seq[:2])
    print('增加维度后:',seq[:, :, np.newaxis].shape)
    print(seq[:2])
    # returned seq, res and xs: shape (batch, step, input)
    #np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

# class LSTMRNN
class LSTMRNN(object):
    def __init__(self,n_steps,input_size,output_size,cell_size,batch_size):
        '''
            :param n_steps: 每批数据总包含多少时间刻度
            :param input_size: 输入数据的维度
            :param output_size: 输出数据的维度 如果是类似价格曲线的话，应该为1
            :param cell_size: cell的大小
            :param batch_size: 每批次训练数据的数量
        '''
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope("inputs"): # xs->seq, ys ->res
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()
        with tf.variable_scope("LSTM-cell"):
            self.add_cell()
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()
        with tf.name_scope("cost"):
            self.compute_cost()
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
    def add_input_layer(self):
        # l_in_x:(batch*n_step, in_size),相当于把这个批次的样本串到一个长度1000的时间线上，每批次50个样本，每个样本20个时刻
#       # (batch*n_step, in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        # Ws in_size, cell_size)
        Ws_in = self._weight_variabe([self.input_size, self.cell_size])
        # bs (cell_size,)
        bs_in = self._biases_variabe([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in)+bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
        #多时刻的状态叠加层
    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial-state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
#           # time_major=False 表示时间主线不是第一列batch
            self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
        lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
    def add_output_layer(self):
        # shape = (batch*steps, cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variabe([self.cell_size, self.output_size])
        bs_out = self._biases_variabe([self.output_size,])
        #shape = (batch*steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out)+bs_out   
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')], 
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)
    def ms_error(self, labels, logits): #参数可能是因为 tf.contrib.legacy_seq2seq.sequence_loss_by_example参数的
        return tf.square(tf.subtract(labels,logits))       
    def _weight_variabe(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)      
    def _biases_variabe(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, initializer=initializer, shape=shape)

if __name__ == '__main__':
    seq, res  = get_batch_data()
 
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(tf.global_variables_initializer())
    # tf.initialize_all_variables() no long valid from
    #sess.run(tf.global_variables_initializer())
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    for j in range(1):#训练200次
        pred_res=None
        for i in range(20):#把整个数据分为20个时间段
            seq, res = get_batch_data()
 
            if i == 0:
                feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state    # use last state as the initial state for this run
                }
 
            _, cost, state, pred = sess.run(
                [model.train_op, model.cost, model.cell_final_state, model.pred],
                feed_dict=feed_dict)
            pred_res=pred
 
 
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
        print('{0} cost: '.format(j ), round(cost, 4))
        BATCH_START=0 #从头再来一遍
 
    # 画图
    print("结果:",pred_res.shape)
    #与最后一次训练所用的数据保持一致
    train_y = train_y[190:490]
    print('实际',train_y.flatten().shape)
 
    r_size=BATCH_SIZE * TIME_STEPS
    ###画图###########################################################################
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    #为了方便看，只显示了后100行数据
    line1,=axes.plot(range(50), pred.flatten()[-50:] , 'b--',label='bp')
    line3,=axes.plot(range(50), train_y.flatten()[ - 50:], 'r',label='bp')
 
    axes.grid()
    fig.tight_layout()
    #plt.legend(handles=[line1, line2,line3])
    #plt.legend(handles=[line1,  line3])
    plt.legend(handles=[line3])
    plt.title('-')
    plt.show()