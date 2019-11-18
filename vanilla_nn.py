import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import sys

class NeuralNetwork:

	def __init__(self):
		self.layers = []
		
		self.derivative_errors = {
		"mse": lambda targets,outputs : -(targets - outputs),
		"crossentropy": lambda targets,outputs : (outputs - targets)
		}

		pass

	def predict(self,input):

		for layer in self.layers:
			input = layer.feedfoward(input)
			pass

		return self.layers[len(self.layers) - 1].output

	def train(self,x_train,y_train,epochs,loss):
		scorecard = []
		for e in range(epochs):
			print('epoch', e)
			for t in range(len(x_train)):						
				#feed foward

				self.predict(x_train[t])

				self.layers.reverse()

				targets = np.zeros(10)
				targets[y_train[t]] = 1

				if np.argmax(self.layers[0].output) == y_train[t]:
					scorecard.append(1)
				else:
					scorecard.append(0)

				if t % 1000 == 0 and t != 0:
					sys.stdout.write(str(t) + '/' + str(len(x_train)) + ' accuracy:' + str(sum(scorecard) / 1000 * 100) + '%' + '\n')
					sys.stdout.flush()
					scorecard.clear()

				targets = np.array(targets,ndmin=2).T

				error = self.derivative_errors[loss](targets,self.layers[0].output)
					
				for layer_i in range(len(self.layers)):
					if self.layers[layer_i].needsWeights:
						next_error = np.dot(self.layers[layer_i].W.T,error)
						
						self.layers[layer_i].backprop(error,loss,layer_i)

						error = next_error
						pass
					pass
					
				self.layers.reverse()
			pass

		pass

	def add(self,layer):
		self.layers.append(layer)
		pass

	def save_model(self):
		pass	
	pass

class dense_layer():

	def __init__(self,nodes,lr,activation = 'linear'):

		self.nodes = nodes

		self.activation_functions = {
		"linear": lambda x : x,
		"sigmoid": lambda x : 1/(1 + np.exp(-x)),
		"relu": lambda x : abs(x * (x > 0)),
		"tanh": lambda x : (2 / (1 + np.exp(-2 * x))) - 1,
		"softmax": lambda x : np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
		}

		self.derivatives = {
		"linear": lambda x : 1,
		"sigmoid": lambda x : x * (1.0 - x),
		"relu": lambda x : abs(1 * (x > 0)),
		"tanh": lambda x : 1.0 - x ** 2,
		"softmax": lambda x : x * (1.0 - x)
		}

		self.activation = activation

		self.activation_function = self.activation_functions[activation]

		self.lr = lr

		self.needsWeights = True

	pass

	def setWeights(self,x,y):

		if self.activation == 'sigmoid' or self.activation == 'tanh' or self.activation == 'softmax':
			self.W = np.random.normal(0.0,pow(1 / float(x), 0.5),(x,y))
		elif self.activation == 'relu':
			self.W = np.random.normal(0.0,pow(2 / float(x), 0.5),(x,y))
		else:
			self.W = np.random.rand(x,y)
	
	pass

	def feedfoward(self,X):

		self.input = X

		if not hasattr(self, 'W'): 
			self.setWeights(self.nodes,self.input.shape[0])

		self.output = self.activation_function(np.dot(self.W,X)) 

		return self.output 	


	def backprop(self,E,loss,layer_i):

		if loss == 'crossentropy' and layer_i == 0:	
			self.W -= self.lr * np.dot(E,np.transpose(self.input))
		else:
			self.W -= self.lr * np.dot(E * self.derivatives[self.activation](self.output),np.transpose(self.input))

		pass

	pass

class flatten:

	def __init__(self):
		self.needsWeights = False		

		pass

	def feedfoward(self,X):

		flatX = np.zeros(X.shape[0] * X.shape[1])
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				flatX[i * X.shape[0] + j] = X[i,j] 
				pass
			pass
			
		self.output = np.array(list(flatX),ndmin=2).T
		self.nodes = flatX.shape[0]

		return self.output

	pass


#training and testing the network 

nn = NeuralNetwork()

nn.add(flatten())
nn.add(dense_layer(128,0.001,activation = "sigmoid"))
nn.add(dense_layer(128,0.001,activation = "sigmoid"))
nn.add(dense_layer(10,0.001, activation = "sigmoid"))

#loading train data and training
"""
path_dataset = 'D:/Dev Projects/Machine Learning/NeuralNetworks/Datasets/vanilla_mnist_dataset/'

x_train = idx2numpy.convert_from_file(path_dataset + 'train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file(path_dataset + 'train-labels.idx1-ubyte')

x_train = (x_train / 255.0 * 0.99) + 0.01

nn.train(x_train,y_train,1,loss='crossentropy')


flatX_train = np.zeros((x_train.shape[0],np.prod(x_train.shape[1:])))
for i in range(x_train.shape[0]):
	flatX_train[i,:] = x_train[i].flatten(order='C')

flatX_train = (flatX_train / 255.0 * 0.99) + 0.01

#loading test data and testing

x_test = idx2numpy.convert_from_file(path_dataset + 't10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file(path_dataset + 't10k-labels.idx1-ubyte')

x_test = (x_test / 255.0 * 0.99) + 0.01

scorecard = []

for t in range(len(x_test)):
	print(nn.predict(x_test[t]))
	if np.argmax(nn.predict(x_test[t])) == y_test[t]:
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass

	pass

scorecard_array = np.asarray(scorecard)

print('performance is: ', scorecard_array.sum() / scorecard_array.size * 100 , '%')
"""