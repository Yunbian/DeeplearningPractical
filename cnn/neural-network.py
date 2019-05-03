"""
    使用多层神经网络实现图像分类

"""

import tensorflow as tf
import os
import pickle
import numpy as np


#########################################   数据处理模块  #############################################

CIFAR_DIR = "./cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))


def load_data(filename):
    """read data from data file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        return data[b'data'], data[b'labels']


# tensorflow.Dataset.
class CifarData:
    def __init__(self, filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)
            all_data.append(data)
            all_labels.append(labels)
        self._data = np.vstack(all_data)
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        print(self._data.shape)
        print(self._labels.shape)

        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        # [0,1,2,3,4,5] -> [5,3,2,4,0,1]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self, batch_size):
        """return batch_size examples as a batch."""
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larger than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels


train_filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d' % i) for i in range(1, 6)]
test_filenames = [os.path.join(CIFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

#########################################   开始神经网络部分  #############################################

# 输入层
x = tf.placeholder(tf.float32, [None, 3072])
# [None], eg: [0,5,6,3]
y = tf.placeholder(tf.int64, [None])

# 隐藏层
hidden1 = tf.layers.dense(x, 100, activation=tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 100, activation=tf.nn.relu)
hidden3 = tf.layers.dense(hidden2, 50, activation=tf.nn.relu)

# 全连接层
y_ = tf.layers.dense(hidden3, 10)


# 损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
# y_ -> sofmax
# y -> one_hot
# loss = ylogy_

# indices，预测值
predict = tf.argmax(y_, 1)
# [1,0,1,1,1,0,0,0]
correct_prediction = tf.equal(predict, y)

# 精确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 优化函数
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


#########################################   网络训练  #############################################

init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 100

# train: 100k: 51.%
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):
        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ = sess.run(
            [loss, accuracy, train_op],
            feed_dict={
                x: batch_data,
                y: batch_labels})
        # 500步输出信息
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f'
                  % (i+1, loss_val, acc_val))

        # 每5000步对测试数据集进行测试
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict = {
                        x: test_batch_data,
                        y: test_batch_labels
                    })
                all_test_acc_val.append(test_acc_val)
            test_acc = np.mean(all_test_acc_val)
            print('[Test ] Step: %d, acc: %4.5f'
                  % (i+1, test_acc))

# [Test ] Step: 100000, acc: 0.49300
