from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import _pickle as pickle
import glob
from PIL import Image
import random
import numpy as np

def batch_norm(input, output_shape, isTraining, scope):
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0,shape=[output_shape]),name='beta',trainable=True)
		gamma = tf.Variable(tf.constant(1.0,shape=[output_shape]),name='gamma',trainable=True)
		batch_mean, batch_variance = tf.nn.moments(input,[0,1,2],name="moments")
		moving_average = tf.train.ExponentialMovingAverage(decay=0.95)
		def mean_var_with_update():
			moving_average_apply_op = moving_average.apply([batch_mean,batch_variance])
			with tf.control_dependencies([moving_average_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_variance)
		mean, var = tf.cond(isTraining,mean_var_with_update,lambda: (moving_average.average(batch_mean),moving_average.average(batch_variance)))
	return tf.nn.batch_normalization(input,mean,var,beta,gamma,1e-3)

# http://www.ais.uni-bonn.de/download/datasets.html
training_dataset_dir="/home/vimal/Downloads/labelme-dataset/LabelMe-12-50k/train/"
test_dataset_dir="/home/vimal/Downloads/labelme-dataset/LabelMe-12-50k/test/"
labels_names=['person','car','building','window','tree','sign','door','bookshelf','chair','table','keyboard','head']
training_labels=[]
training_data=[]
lines=[]

# Train data
with open("/home/vimal/Downloads/labelme-dataset/LabelMe-12-50k/train/annotation.txt") as file:
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
with open("/home/vimal/Downloads/labelme-dataset/LabelMe-12-50k/test/annotation.txt") as file:
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

# Training
training_data = np.array([np.array(Image.open(fname)) for fname in training_data],dtype=np.float32)
training_labels = np.reshape(training_labels,[len(training_labels),12])

# Test
test_data = np.array([np.array(Image.open(fname)) for fname in test_data],dtype=np.float32)
test_labels = np.reshape(test_labels,[len(test_labels),12])


inputs_data=tf.placeholder(tf.float32, [None,256,256,3])
labels=tf.placeholder(tf.int64, [None,12])
isTraining = tf.placeholder(tf.bool)
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
pooling_size = [1,3,3,1]
pooling_stride = [1,2,2,1]

# Convolutional Layer #1
with tf.name_scope("conv_layer_1"):
	filter1 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,stddev=1e-1), name='weights_1')
	conv1 = tf.nn.conv2d(inputs_data,filter1,[1,1,1,1],padding="SAME")
	batch_norm_1 = batch_norm(conv1,64,isTraining,"batch_1")
	relu1 = tf.nn.relu(batch_norm_1)

# Convolutional Layer #2
layer_name = "_2"
pool_name="_1"
with tf.name_scope("conv_layer"+layer_name):
	filter2 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv2 = tf.nn.conv2d(relu1,filter2,[1,1,1,1],padding="SAME")
	batch_norm_2 = batch_norm(conv2,128,isTraining,"batch"+layer_name)
	relu2 = tf.nn.relu(batch_norm_2)
with tf.name_scope("pool_layer"+layer_name):
	pool2 = tf.nn.max_pool(relu2, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)

# Convolutional Layer #3
layer_name = "_3"
pool_name="_2"
with tf.name_scope("conv_layer"+layer_name):
	filter3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv3 = tf.nn.conv2d(pool2,filter3,[1,1,1,1],padding="SAME")
	batch_norm_3 = batch_norm(conv3,128,isTraining,"batch"+layer_name)
	relu3 = tf.nn.relu(batch_norm_3)
with tf.name_scope("pool_layer"+layer_name):
	pool3 = tf.nn.max_pool(relu3, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)

# Convolutional Layer #4
layer_name = "_4"
pool_name="_3"
with tf.name_scope("conv_layer"+layer_name):
	filter4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv4 = tf.nn.conv2d(pool3,filter4,[1,1,1,1],padding="SAME")
	batch_norm_4 = batch_norm(conv4,128,isTraining,"batch"+layer_name)
	relu4 = tf.nn.relu(batch_norm_4)
with tf.name_scope("pool_layer"+layer_name):
	pool4 = tf.nn.max_pool(relu4, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)

# Convolutional Layer #5
layer_name = "_5"
with tf.name_scope("conv_layer"+layer_name):
	filter5 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv5 = tf.nn.conv2d(pool4,filter5,[1,1,1,1],padding="SAME")
	batch_norm_5 = batch_norm(conv5,128,isTraining,"batch"+layer_name)
	relu5 = tf.nn.relu(batch_norm_5)

# Convolutional Layer #6
layer_name = "_6"
with tf.name_scope("conv_layer"+layer_name):
	filter6 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv6 = tf.nn.conv2d(relu5,filter6,[1,1,1,1],padding="SAME")
	batch_norm_6 = batch_norm(conv6,128,isTraining,"batch"+layer_name)
	relu6 = tf.nn.relu(batch_norm_6)

# Convolutional Layer #7
layer_name = "_7"
pool_name="_4"
with tf.name_scope("conv_layer"+layer_name):
	filter7 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv7 = tf.nn.conv2d(relu6,filter7,[1,1,1,1],padding="SAME")
	batch_norm_7 = batch_norm(conv7,128,isTraining,"batch"+layer_name)
	relu7 = tf.nn.relu(batch_norm_7)
with tf.name_scope("pool_layer"+layer_name):
	pool7 = tf.nn.max_pool(relu7, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)

# Convolutional Layer #8
layer_name = "_8"
pool_name="_5"
with tf.name_scope("conv_layer"+layer_name):
	filter8 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv8 = tf.nn.conv2d(pool7,filter8,[1,1,1,1],padding="SAME")
	batch_norm_8 = batch_norm(conv8,128,isTraining,"batch"+layer_name)
	relu8 = tf.nn.relu(batch_norm_8)
with tf.name_scope("pool_layer"+layer_name):
	pool8 = tf.nn.max_pool(relu8, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)

# Convolutional Layer #9
layer_name = "_9"
pool_name="_6"
with tf.name_scope("conv_layer"+layer_name):
	filter9 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv9 = tf.nn.conv2d(pool8,filter9,[1,1,1,1],padding="SAME")
	batch_norm_9 = batch_norm(conv9,128,isTraining,"batch"+layer_name)
	relu9 = tf.nn.relu(batch_norm_9)
with tf.name_scope("pool_layer"+layer_name):
	pool9 = tf.nn.max_pool(relu9, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)
	pool9_dropout = tf.nn.dropout(pool9, 0.5)

# Convolutional Layer #10
layer_name = "_10"
with tf.name_scope("conv_layer"+layer_name):
	filter10 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv10 = tf.nn.conv2d(pool9_dropout,filter10,[1,1,1,1],padding="SAME")
	batch_norm_10 = batch_norm(conv10,128,isTraining,"batch"+layer_name)
	relu10 = tf.nn.relu(batch_norm_10)

# Convolutional Layer #11
layer_name = "_11"
with tf.name_scope("conv_layer"+layer_name):
	filter11 = tf.Variable(tf.truncated_normal([1, 1, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv11 = tf.nn.conv2d(relu10,filter11,[1,1,1,1],padding="SAME")
	batch_norm_11 = batch_norm(conv11,128,isTraining,"batch"+layer_name)
	relu11 = tf.nn.relu(batch_norm_11)

# Convolutional Layer #12
layer_name = "_12"
pool_name="_7"
with tf.name_scope("conv_layer"+layer_name):
	filter12 = tf.Variable(tf.truncated_normal([1, 1, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv12 = tf.nn.conv2d(relu11,filter12,[1,1,1,1],padding="SAME")
	batch_norm_12 = batch_norm(conv12,128,isTraining,"batch"+layer_name)
	relu12 = tf.nn.relu(batch_norm_12)
with tf.name_scope("pool_layer"+layer_name):
	pool12 = tf.nn.max_pool(relu12, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)

# Convolutional Layer #13
layer_name = "_13"
pool_name="_8"
with tf.name_scope("conv_layer"+layer_name):
	filter13 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,stddev=1e-1), name="weights"+layer_name)
	conv13 = tf.nn.conv2d(pool12,filter13,[1,1,1,1],padding="SAME")
with tf.name_scope("pool_layer"+layer_name):
	pool13 = tf.nn.max_pool(conv13, ksize=pooling_size,strides=pooling_stride,padding="SAME",name="pool"+pool_name)
	pool13_dropout = tf.nn.dropout(pool13, 0.5)

# Logits Layer
with tf.name_scope("softmax_layer"):
	shape = int(np.prod(pool13_dropout.get_shape()[1:]))
	dropout_flat = tf.reshape(pool13_dropout, [-1,shape])
	fc3w = tf.Variable(tf.truncated_normal([128, 12], dtype=tf.float32,stddev=1e-1), name='fc3_weights')
	fc3b = tf.Variable(tf.constant(1.0, shape=[12], dtype=tf.float32),trainable=True, name='fc3_biases')
	fc3out = tf.nn.bias_add(tf.matmul(dropout_flat, fc3w), fc3b)
	logits  = tf.nn.softmax(fc3out)

with tf.name_scope("cross_entropy_layer"):
	# Calculate Loss
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
	loss = tf.reduce_mean(cross_entropy)

with tf.name_scope("training_layer"):
	# Optimize Loss
	train_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss,global_step=global_step)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
# Parameter definitions
mini_batch_size = 10
epochs = 10000

n = len(training_labels)
indices = np.arange(n,dtype=np.int32)
np.random.shuffle(indices)

n_test = len(test_labels)
indices_test = np.arange(n_test,dtype=np.int32)
np.random.shuffle(indices_test)

train_writer = tf.summary.FileWriter("./logs/5/train",session.graph)
test_writer = tf.summary.FileWriter("./logs/5/test")
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
tf.summary.scalar("weights1", tf.reduce_mean(filter1))
tf.summary.scalar("weights14", tf.reduce_mean(fc3w))
#tf.summary.image("layer_1",filter1,max_outputs=3)
#tf.summary.image("layer_13",pool13,max_outputs=3)
write_op = tf.summary.merge_all()

step = 0
# Training
for j in range(epochs):
	mini_batches = [indices[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
	test_mini_batches = [indices_test[k:k+mini_batch_size] for k in range(0, n_test, mini_batch_size)]
	accuracy_history = []
	for mini_batch in mini_batches:
		batch_xs = training_data[mini_batch]
		batch_ys = training_labels[mini_batch]

		session.run(train_op,feed_dict={inputs_data: batch_xs, labels: batch_ys, isTraining: True})
		if step % 10 == 0:
			summary1 = session.run(write_op,feed_dict={inputs_data: batch_xs, labels: batch_ys, isTraining: True})
			train_writer.add_summary(summary1, step)
			train_writer.flush()

			summary2 = session.run(write_op,feed_dict={inputs_data: test_data[test_mini_batches[0]], labels: test_labels[test_mini_batches[0]], isTraining: False})
			test_writer.add_summary(summary2, step)
			test_writer.flush()
		step = step + 1
	for test_mini_batch in test_mini_batches:
		acc = session.run(accuracy,feed_dict={inputs_data: test_data[test_mini_batch], labels: test_labels[test_mini_batch], isTraining: False})
		accuracy_history.append(acc)
	print('Epoch number {} Training Accuracy: {}'.format(j+1, np.mean(accuracy_history)))
	if (j+1) % 50 == 0:
		saver.save(session, "./logs/5/ckpt/model",global_step=50)
