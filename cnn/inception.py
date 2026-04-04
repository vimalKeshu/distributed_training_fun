from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import _pickle as pickle
import glob
from PIL import Image
import random
import numpy as np
import os


tf.reset_default_graph()

num_classes = 10
num_channels = 3
img_size = 32
images_per_file = 10000
num_images_train = 5 * images_per_file
cifar1_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
weight_decay=0.00004

def convert_images(raw):
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    return images

def convert_label(raw_labels):
	labels = []
	for label in raw_labels:
		label_list=[0,0,0,0,0,0,0,0,0,0]
		label_list[label] = 1
		labels.append(label_list)
	return np.array(labels)

def inception_block(inputs,filter_size,isTraining):
	with tf.variable_scope('Branch_0'):
		branch_0_conv = tf.layers.conv2d(inputs=inputs,filters=filter_size,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),kernel_size=[1, 1],use_bias=False,padding="same")
		branch_0_batch_norm = tf.layers.batch_normalization(branch_0_conv, training=isTraining)
		branch_0 = tf.nn.relu(branch_0_batch_norm)
	with tf.variable_scope('Branch_1'):
		branch_1a_conv = tf.layers.conv2d(inputs=inputs,filters=filter_size,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),kernel_size=[1, 1],use_bias=False,padding="same")
		branch_1a_batch_norm = tf.layers.batch_normalization(branch_1a_conv, training=isTraining)
		branch_1a = tf.nn.relu(branch_1a_batch_norm)

		branch_1b_conv = tf.layers.conv2d(inputs=branch_1a,filters=filter_size+32,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),kernel_size=[3, 3],use_bias=False,padding="same")
		branch_1b_batch_norm = tf.layers.batch_normalization(branch_1b_conv, training=isTraining)
		branch_1 = tf.nn.relu(branch_1b_batch_norm)
	with tf.variable_scope('Branch_2'):
		branch_2a_conv = tf.layers.conv2d(inputs=inputs,filters=filter_size,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),kernel_size=[1, 1],use_bias=False,padding="same")
		branch_2a_batch_norm = tf.layers.batch_normalization(branch_2a_conv, training=isTraining)
		branch_2a = tf.nn.relu(branch_2a_batch_norm)

		branch_2b_conv = tf.layers.conv2d(inputs=branch_2a,filters=filter_size+32,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),kernel_size=[5, 5],use_bias=False,padding="same")
		branch_2b_batch_norm = tf.layers.batch_normalization(branch_2b_conv, training=isTraining)
		branch_2 = tf.nn.relu(branch_2b_batch_norm)
	with tf.variable_scope('Branch_3'):
		branch_3_max_pool = tf.layers.max_pooling2d(inputs=inputs, pool_size=[3, 3], strides=1, padding="same")

		branch_3_conv = tf.layers.conv2d(inputs=branch_3_max_pool,filters=filter_size+32,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),kernel_size=[1, 1],use_bias=False,padding="same")
		branch_3_batch_norm = tf.layers.batch_normalization(branch_3_conv, training=isTraining)
		branch_3 = tf.nn.relu(branch_3_batch_norm)
	return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

# cifar10
dataset_dir="/home/vimal/Downloads/cifar-10-batches-py/"
log_dir = "./logs/"

train_label=np.zeros(shape=[num_images_train, num_classes], dtype=np.int32)
train_data=np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=np.float32)

test_label=np.zeros(shape=[images_per_file, num_classes], dtype=np.int32)
test_data=np.zeros(shape=[images_per_file, img_size, img_size, num_channels], dtype=np.float32)

# Train data
begin = 0
for i in range(5):
	with open(dataset_dir+"data_batch_"+str(i+1),'rb') as file:
		datapack = pickle.load(file, encoding='bytes')
		images = convert_images(datapack[b'data'])
		labels = convert_label(datapack[b'labels'])
		end = begin + len(images)
		train_data[begin:end,:] = images
		train_label[begin:end,:] = labels
		begin = end

begin = 0
with open(dataset_dir+"test_batch",'rb') as file:
	datapack = pickle.load(file, encoding='bytes')
	images = convert_images(datapack[b'data'])
	labels = convert_label(datapack[b'labels'])
	end = begin + len(images)
	test_data[begin:end,:] = images
	test_label[begin:end,:] = labels


inputs_data=tf.placeholder(tf.float32, [None, img_size, img_size, 3])
labels=tf.placeholder(tf.int64, [None, num_classes])
isTraining = tf.placeholder(tf.bool)
dropout_rate = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
filter_size = 64


with tf.variable_scope("inception"):
	with tf.variable_scope('Base_layer'):
		net_conv = tf.layers.conv2d(inputs=inputs_data,filters=filter_size,kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_size=[1, 1],use_bias=False,padding="same")
		net_batch_norm = tf.layers.batch_normalization(net_conv, training=isTraining)
		net = tf.nn.relu(net_batch_norm)

	for idx in range(4):
		with tf.variable_scope('inception_block'+str(idx+1)):
			net = inception_block(net,filter_size,isTraining)
			net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3], strides=2)
			filter_size = filter_size + 64

	print(net.get_shape())
	shape = int(np.prod(net.get_shape()[1:]))
	layer_flat = tf.reshape(net, [-1,shape])
	dense = tf.layers.dense(inputs=layer_flat, units=shape, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate, training=isTraining)
	output = tf.layers.dense(inputs=dropout, units=num_classes)
	logits  = tf.nn.softmax(output)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		with tf.variable_scope('cross_entropy_layer'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
			loss = tf.reduce_mean(cross_entropy)
		with tf.variable_scope('training_layer'):
			train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

	with tf.variable_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()


ckpt = tf.train.get_checkpoint_state(os.path.dirname(log_dir))
if ckpt and ckpt.model_checkpoint_path:
	saver.restore(session,ckpt.model_checkpoint_path)
	print("restored the model...")

mini_batch_size = 50
epochs = 1000

n = len(train_data)
indices = np.arange(n,dtype=np.int32)
np.random.shuffle(indices)

n_test = len(test_data)
indices_test = np.arange(n_test,dtype=np.int32)
np.random.shuffle(indices_test)

step = 0
# Training
for j in range(epochs):
	mini_batches = [indices[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
	test_mini_batches = [indices_test[k:k+mini_batch_size] for k in range(0, n_test, mini_batch_size)]
	accuracy_history = []
	train_accuracy_history = []
	for index,mini_batch in enumerate(mini_batches):
		batch_xs = train_data[mini_batch]
		batch_ys = train_label[mini_batch]
		session.run(train_op,feed_dict={inputs_data: batch_xs, labels: batch_ys, isTraining: True, dropout_rate: 0.5,learning_rate: 0.0001})
		if (index + 1) % 50 == 0:
			acc = session.run(accuracy,feed_dict={inputs_data: batch_xs, labels: batch_ys, isTraining: False, dropout_rate: 1, learning_rate: 0.0001})
			train_accuracy_history.append(acc)
	print('Epoch number {} Training Accuracy: {}'.format(j+1, np.mean(train_accuracy_history)))
	for test_mini_batch in test_mini_batches:
		acc = session.run(accuracy,feed_dict={inputs_data: test_data[test_mini_batch], labels: test_label[test_mini_batch], isTraining: False, dropout_rate: 1, learning_rate: 0.0001})
		accuracy_history.append(acc)
	print('Epoch number {} Test Accuracy: {}'.format(j+1, np.mean(accuracy_history)))
	if (j+1) % 50 == 0:
		saver.save(session,log_dir+"/model",global_step=50)
