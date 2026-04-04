import tensorflow as tf
import _pickle as pickle
import gzip
import random
import numpy as np

def variable_summaries(var):
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

with gzip.open('/home/vimal/Downloads/neural-networks-and-deep-learning/data/mnist.pkl.gz','rb') as file:
	training_data, validation_data, test_data = pickle.load(file,encoding='latin1')

if len(training_data) == 0 or len(training_data[0]) != len(training_data[1]):
	raise ValueError('Mnist input and labels size are not same.')

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.int64, [None])

with tf.name_scope("layer_1"):
	with tf.name_scope("weights"):
		w_layer_1 = tf.Variable(tf.zeros([784,500]))
		variable_summaries(w_layer_1)
	with tf.name_scope("biases"):
		b_layer_1 = tf.Variable(tf.zeros([500]))
		variable_summaries(b_layer_1)
	with tf.name_scope("matmul"):
		y_layer_1 = tf.matmul(x,w_layer_1) + b_layer_1
		variable_summaries(y_layer_1)
	with tf.name_scope("activation"):
		z_layer_1 = tf.nn.sigmoid(y_layer_1)
		variable_summaries(z_layer_1)

with tf.name_scope("layer_2"):
	with tf.name_scope("weights"):
		w_layer_2 = tf.Variable(tf.zeros([500,10]))
		variable_summaries(w_layer_2)
	with tf.name_scope("biases"):
		b_layer_2 = tf.Variable(tf.zeros([10]))
		variable_summaries(b_layer_2)
	with tf.name_scope("matmul"):
		y_layer_2 = tf.matmul(z_layer_1,w_layer_2) + b_layer_2
		variable_summaries(y_layer_2)
#z_layer_2 = tf.nn.sigmoid(y_layer_2)

#cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=z_layer_2)
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
with tf.name_scope('cross_entropy'):
	cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y_layer_2)
	tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
	# Test trained model
	correct_prediction = tf.equal(tf.argmax(z_layer_1, 1), y_)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

epochs = 1000
mini_batch_size = 100
n = len(training_data[0])
indices = np.arange(n)
np.random.shuffle(indices)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs/1/train',session.graph)
test_writer = tf.summary.FileWriter('./logs/1/test')

for j in range(epochs):
	mini_batches = [indices[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
	for mini_batch in mini_batches:
		batch_xs = training_data[0][mini_batch]
		batch_ys = training_data[1][mini_batch]
		summary, _ = session.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
		train_writer.add_summary(summary, j)
	if j % 10 == 0:
		summary, acc = session.run([merged, accuracy], feed_dict={x: test_data[0],y_: test_data[1]})
		test_writer.add_summary(summary, j)
		print('Accuracy at step %s: %s' % (j, acc))

summary, acc = session.run([merged, accuracy], feed_dict={x: test_data[0],y_: test_data[1]})
test_writer.add_summary(summary, epochs)
print('Accuracy at step %s: %s' % (epochs, acc))
