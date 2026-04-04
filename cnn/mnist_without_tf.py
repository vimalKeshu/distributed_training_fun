import _pickle as pickle
import gzip
import random
import numpy as np

def vectorized_labels(label_index):
	labels = np.zeros((10,1))
	labels[label_index] = 1.0
	return labels

def evaluate(test_data):
	pass

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(output_activations, y):
    return (output_activations-y)


def feed_forward(input,weights,biases):
	activation = input
	activations = [input]
	zs = []
	for b , w in zip(biases,weights):
		z = np.dot(w,activation) + b
		zs.append(z)
		activation = sigmoid(z)
		activations.append(activation)
	return (zs,activations)

def feed_backward(zs,activations,weights,biases,output,num_layers):
	nabla_b = [np.zeros(b.shape) for b in biases]
	nabla_w = [np.zeros(w.shape) for w in weights]

	delta = cost_derivative(activations[-1],output) * sigmoid_prime(zs[-1])
	nabla_b[-1] = delta
	nabla_w[-1] = np.dot(delta, activations[-2].transpose())
	for l in range(2,num_layers):
		z = zs[-l]
		sp = sigmoid_prime(z)
		delta = np.dot(weights[-l+1].transpose(), delta) * sp
		nabla_b[-l] = delta
		nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
	return (nabla_b,nabla_w)

def update_mini_batch(mini_batch, weights, biases, num_layers, eta):
	nabla_b = [np.zeros(b.shape) for b in biases]
	nabla_w = [np.zeros(w.shape) for w in weights]
	for input, output in mini_batch:
		z_sigmoid, activations = feed_forward(input,weights,biases)
		delta_nabla_b, delta_nabla_w = feed_backward(z_sigmoid, activations, weights, biases, output, num_layers)
		nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
	weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(weights, nabla_w)]
	biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(biases, nabla_b)]
	return (weights, biases)

def SGD(training_data, epochs, mini_batch_size, eta, weights, biases,num_layers, test_data=None):
	if test_data: n_test = len(test_data)
	training_data = list(training_data)
	n = len(training_inputs)
	for j in range(epochs):
		random.shuffle(training_data)
		mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
		for mini_batch in mini_batches:
			weights, biases = update_mini_batch(mini_batch, weights, biases, num_layers, eta)
		if test_data:
			print("Epoch {0}: {1} / {2}",j, evaluate(test_data), n_test)
		else:
			print("Epoch {0} complete",j)
	return (weights, biases)

with gzip.open('/home/vimal/Downloads/neural-networks-and-deep-learning/data/mnist.pkl.gz','rb') as file:
	training_data, validation_data, test_data = pickle.load(file,encoding='latin1')

training_inputs = [np.reshape(x,(784,1)) for x in training_data[0]]
training_labels = [vectorized_labels(y) for y in training_data[1]]

layers = [784,20,10]
num_layers = len(layers)
biases = [np.random.randn(y,1) for y in layers[1:]]
weights = [ np.random.randn(y,x) for x,y in zip(layers[:-1],layers[1:])]

weights, biases = SGD(zip(training_inputs,training_labels), 1000, 100, 0.01, weights, biases, num_layers)

print(weights)
print(biases)
print("test")
