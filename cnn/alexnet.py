from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import _pickle as pickle
import glob
from PIL import Image
import random
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

# http://www.ais.uni-bonn.de/download/datasets.html
training_dataset_dir="/home/vimal/Downloads/labelme-dataset/LabelMe-12-50k/train/"
test_dataset_dir="/home/vimal/Downloads/labelme-dataset/LabelMe-12-50k/test/"
labels_names=['person','car','building','window','tree','sign','door','bookshelf','chair','table','keyboard','head']
log_dir = "./logs/1/"
training_labels=[]
training_data=[]
lines=[]

# Train data
with open(training_dataset_dir+"annotation.txt") as file:
	lines = [line.split() for line in file.readlines()]

counter = 0
for line in lines:
	file_path = training_dataset_dir+"0"+line[0][:3]+"/"+line[0]+".jpg"
	for index,label in enumerate(line[1:]):
		l = float(label)
		if l >= 0.00:
			lablels_list=[0,0,0,0,0,0,0,0,0,0,0,0]
			lablels_list[index]=1
			training_labels.insert(counter,lablels_list)
			training_data.insert(counter,file_path)
			counter = counter + 1

# Test data
with open(test_dataset_dir+"annotation.txt") as file:
	lines = [line.split() for line in file.readlines()]

test_labels=[]
test_data=[]
counter = 0
for line in lines:
	file_path = training_dataset_dir+"0"+line[0][:3]+"/"+line[0]+".jpg"
	for index,label in enumerate(line[1:]):
		l = float(label)
		if l >= 0.00:
			lablels_list=[0,0,0,0,0,0,0,0,0,0,0,0]
			lablels_list[index]=1
			test_labels.insert(counter,lablels_list)
			test_data.insert(counter,file_path)
			counter = counter + 1

print("total lines : ",len(lines))
print("labels : ",len(training_labels))
print("data : ",len(training_data))
print("test labels : ",len(test_labels))
print("test data : ",len(test_data))

training_labels = np.asarray(training_labels,dtype=np.int32)
training_data = np.asarray(training_data)

# Test
test_batch_xs = np.array([np.array(Image.open(fname)) for fname in test_data],dtype=np.float32)
test_batch_ys = np.reshape(test_labels,[len(test_labels),12])


inputs_data=tf.placeholder(tf.float32, [None,256,256,3])
labels=tf.placeholder(tf.int64, [None,12])
dropout_rate = tf.placeholder(tf.float32)

# Convolutional Layer #1
with tf.name_scope("conv_layer_1"):
	filter1_1 = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,stddev=1e-1), name='weights1_1')
	biases1 = tf.Variable(tf.constant(0.0,shape=[96],dtype=tf.float32),trainable=True,name="biases1_1")
	stride = [1,4,4,1]
	conv1 = tf.nn.conv2d(inputs_data,filter1_1,stride,padding="SAME")
	out1 = tf.nn.bias_add(conv1,biases1)
	conv1_1 = tf.nn.relu(out1)
with tf.name_scope("pool_layer_1"):
	pool1 = tf.nn.max_pool(conv1_1, ksize=[1, 3, 3, 1],strides=[1,2,2,1],padding="SAME",name="pool1_1")
with tf.name_scope("norm_layer_1"):
	norm1 = tf.nn.local_response_normalization(pool1)

# Convolutional Layer #2
with tf.name_scope("conv_layer_2"):
	filter2_1 = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,stddev=1e-1), name='weights2_1')
	biases2 = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name="biases2_1")
	conv2 = tf.nn.conv2d(norm1,filter2_1,[1,1,1,1],padding="SAME")
	out2 = tf.nn.bias_add(conv2,biases2)
	conv2_1 = tf.nn.relu(out2)
with tf.name_scope("pool_layer_2"):
	pool2 = tf.nn.max_pool(conv2_1, ksize=[1, 3, 3, 1],strides=[1,2,2,1],padding="SAME",name="pool2_1")
with tf.name_scope("norm_layer_2"):
	norm2 = tf.nn.local_response_normalization(pool2)

# Convolutional Layer #3
with tf.name_scope("conv_layer_3"):
	filter3_1 = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32,stddev=1e-1), name='weights3_1')
	biases3 = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name="biases3_1")
	conv3 = tf.nn.conv2d(norm2,filter3_1,[1,1,1,1],padding="SAME")
	out3 = tf.nn.bias_add(conv3,biases3)
	conv3_1 = tf.nn.relu(out3)

# Convolutional Layer #4
with tf.name_scope("conv_layer_4"):
	filter4_1 = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32,stddev=1e-1), name='weights4_1')
	biases4 = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name="biases4_1")
	conv4 = tf.nn.conv2d(conv3_1,filter4_1,[1,1,1,1],padding="SAME")
	out4 = tf.nn.bias_add(conv4,biases4)
	conv4_1 = tf.nn.relu(out4)

# Convolutional Layer #5
with tf.name_scope("conv_layer_5"):
	filter5_1 = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32,stddev=1e-1), name='weights5_1')
	biases5 = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name="biases5_1")
	conv5 = tf.nn.conv2d(conv4_1,filter5_1,[1,1,1,1],padding="SAME")
	out5 = tf.nn.bias_add(conv5,biases5)
	conv5_1 = tf.nn.relu(out5)
with tf.name_scope("pool_layer_3"):
	pool3 = tf.nn.max_pool(conv5_1, ksize=[1, 3, 3, 1],strides=[1,2,2,1],padding="SAME",name="pool3_1")

# Dropout layer
with tf.name_scope("dropout_layer"):
	dropout = tf.nn.dropout(pool3,dropout_rate)

# Dense Layer 1
with tf.name_scope("fc_1") as scope:
	shape = int(np.prod(dropout.get_shape()[1:]))
	fc1w = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32,stddev=1e-1), name='fc1_weights')
	fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),trainable=True, name='fc1_biases')
	dropout_flat = tf.reshape(dropout, [-1,shape])
	fc1out = tf.nn.bias_add(tf.matmul(dropout_flat, fc1w), fc1b)
	fc1 = tf.nn.relu(fc1out)

# Dense Layer 2
with tf.name_scope("fc_2") as scope:
    fc2w = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32,stddev=1e-1), name='fc2_weights')
    fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),trainable=True, name='fc2_biases')
    fc2out = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
    fc2 = tf.nn.relu(fc2out)

# Logits Layer
with tf.name_scope("softmax_layer"):
	fc3w = tf.Variable(tf.truncated_normal([4096, 12], dtype=tf.float32,stddev=1e-1), name='fc3_weights')
	fc3b = tf.Variable(tf.constant(1.0, shape=[12], dtype=tf.float32),trainable=True, name='fc3_biases')
	fc3out = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
	logits  = tf.nn.softmax(fc3out)

with tf.name_scope("cross_entropy_layer"):
	# Calculate Loss
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	loss = tf.reduce_mean(cross_entropy)

with tf.name_scope("training_layer"):
	# Optimize Loss
	train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Parameter definitions
mini_batch_size = 100
epochs = 1000

n = len(training_labels)
indices = np.arange(n,dtype=np.int32)
np.random.shuffle(indices)

train_writer = tf.summary.FileWriter(log_dir+"train",session.graph)
test_writer = tf.summary.FileWriter(log_dir+"test")
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
tf.summary.histogram("weights1_1", filter1_1)
tf.summary.histogram("weights2_1", filter2_1)
tf.summary.histogram("weights3_1", filter3_1)
tf.summary.histogram("weights4_1", filter4_1)
tf.summary.histogram("weights5_1", filter5_1)
write_op = tf.summary.merge_all()

step = 0
# Training
for j in range(epochs):
	accuracy_history = []
	mini_batches = [indices[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
	for mini_batch in mini_batches:
		batch_xs = np.array([np.array(Image.open(fname)) for fname in training_data[mini_batch]],dtype=np.float32)
		batch_ys = np.reshape(training_labels[mini_batch],[len(mini_batch),12])
		acc = session.run([accuracy],feed_dict={inputs_data: batch_xs, labels: batch_ys,dropout_rate:1})
		accuracy_history.append(acc)
		if step % 5 == 0:
			summary1 = session.run(write_op,feed_dict={inputs_data: batch_xs, labels: batch_ys,dropout_rate:1})
			train_writer.add_summary(summary1, step)
			train_writer.flush()

			test_mini_batch = [ix for ix in mini_batch if ix < len(test_data)]
			summary2 = session.run(write_op,feed_dict={inputs_data: test_batch_xs[test_mini_batch], labels: test_batch_ys[test_mini_batch],dropout_rate:1})
			test_writer.add_summary(summary2, step)
			test_writer.flush()
		session.run(train_op,feed_dict={inputs_data: batch_xs, labels: batch_ys,dropout_rate:0.5})
		step = step + 1
	print('Epoch number {} Training Accuracy: {}'.format(j+1, np.mean(accuracy_history)))

print("Test Accuracy",session.run(accuracy,feed_dict={inputs_data: batch_xs, labels: batch_ys,dropout_rate:1}))
