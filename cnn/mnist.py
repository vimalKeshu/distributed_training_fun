#!/usr/bin/python
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/vimal/Downloads/mnist/MNIST_data/",one_hot=True)
import tensorflow as tf

x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x,W) + b
y_ = tf.placeholder(tf.float32, [None, 10])
#cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=y)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
