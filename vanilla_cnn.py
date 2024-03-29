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
					input = x_train[t]
					
					if len(input.shape) == 2:
						input = input.reshape(1,input.shape[0],input.shape[1])
					
					print(np.argmax(self.predict(input)))
					#print(np.argmax(test))

					self.layers.reverse()	

					targets = np.zeros(10)
					targets[y_train[t]] = 1	

					if np.argmax(self.layers[0].output) == y_train[t]:
						scorecard.append(1)
					else:
						scorecard.append(0)

					if t % 10 == 0 and t != 0:
						sys.stdout.write(str(t) + '/' + str(len(x_train)) + ' accuracy:' + str(sum(scorecard) / 10 * 100) + '%' + '\n')
						sys.stdout.flush()
						scorecard.clear()


					targets = np.array(targets,ndmin=2).T

					error = self.derivative_errors[loss](targets,self.layers[0].output)

					for layer_i in range(len(self.layers)):
						error = self.layers[layer_i].backprop(error,loss,layer_i)
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

class dense_layer:

	def __init__(self,nodes,lr,activation = 'linear'):

		self.nodes = nodes

		self.activation_functions = {
		"linear": lambda x : x,
		"sigmoid": lambda x : 1/(1 + np.exp(-x)),
		"relu": lambda x : x * (x > 0),
		"tanh": lambda x : (2 / (1 + np.exp(-2 * x))) - 1,
		"softmax": lambda x : np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
		}

		self.derivatives = {
		"linear": lambda x : 1,
		"sigmoid": lambda x : x * (1.0 - x),
		"relu": lambda x : abs(1 * (x > 0)),
		"tanh": lambda x : 1.0 - x ** 2,
		"softmax": lambda x : x * (1.0 - x)
		}

		self.activation_function = self.activation_functions[activation]

		self.activation = activation

		self.lr = lr

		self.needsWeights = True

	pass

	def setWeights(self,x,y):

		if self.activation == 'sigmoid' or self.activation == 'tanh' or self.activation == 'softmax' or self.activation == 'linear':
			self.W = np.random.normal(0.0,pow(1 / float(x), 0.5),(x,y))
		elif self.activation == 'relu':
			self.W = np.random.normal(0.0,pow(2 / float(x), 0.5),(x,y))
	
	pass

	def feedfoward(self,X):

		self.input = X

		if not hasattr(self, 'W'): 
			self.setWeights(self.nodes,self.input.shape[0])

		self.output = self.activation_function(np.dot(self.W,X)) 

		return self.output


	def backprop(self,E,loss,layer_i):

		next_error = np.dot(self.W.T,E)

		if loss == 'crossentropy' and layer_i == 0:	
			self.W -= self.lr * np.dot(E,np.transpose(self.input))
		else:
			self.W -= self.lr * np.dot(E * self.derivatives[self.activation](self.output),np.transpose(self.input))

		pass

		return next_error

	pass

class conv_layer:

	def __init__(self, filters, filter_size : tuple, strides = (1,1), padding = 'valid',activation='linear', bias = 0.1,lr = 0.001):
		
		self.filters_num = filters
		self.filter_size = filter_size
		self.strides = strides
		self.padding = padding

		self.activation_functions = {
		"linear": lambda x : x,
		"sigmoid": lambda x : 1/(1 + np.exp(-x)),
		"relu": lambda x : x * (x > 0),
		"tanh": lambda x : (2 / (1 + np.exp(-2 * x))) - 1,
		"softmax": lambda x : np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()
		}

		self.derivatives = {
		"linear": lambda x : 1,
		"sigmoid": lambda x : x * (1.0 - x),
		"relu": lambda x : abs(1 * (x > 0)),
		"tanh": lambda x : 1.0 - x ** 2,
		"softmax": lambda x : x * (1.0 - x)
		}

		self.activation_function = self.activation_functions[activation]
		self.activation = activation
		self.bias = bias
		self.lr = lr
		self.filters = np.random.uniform(low = -1, high = 1, size = (filters,filter_size[0],filter_size[1]))
		
		self.needsWeights = True
		
		pass

	def feedfoward(self,X):

		self.input = X

		self.input_shape = X.shape

		image_layers = X.shape[0] 

		image_dim_i = X.shape[1]
		image_dim_j = X.shape[2]

		if self.padding == 'same':
			output_dim_i = image_dim_i
			output_dim_j = image_dim_j

		elif self.padding == 'valid':
			output_dim_i = int((image_dim_i - self.filter_size[0]) / self.strides[0] + 1)
			output_dim_j = int((image_dim_j - self.filter_size[1]) / self.strides[1] + 1)
			pass

		self.output = np.zeros((self.filters_num,output_dim_i,output_dim_j))
		
		for filter_i in range(self.filters_num):
			current_filter = self.filters[filter_i]
			for image_i in range(image_layers):
				self.output[filter_i] += self.convolve2d(X[image_i],current_filter,(output_dim_i,output_dim_j))
				pass

			pass

		self.output += self.bias

		self.output = self.activation_function(self.output)

		return self.output

		pass

	def backprop(self,E,loss,layer_i):

		next_error = np.zeros(self.input_shape)

		for error_filter_i in range(next_error.shape[0]):
			for filter_i in range(self.filters_num):
				W = self.dilate(E[filter_i],(self.strides[0] - 1,self.strides[1] - 1))
				W = self.zeropad(W,((self.filter_size[0] - 1) * 2,(self.filter_size[1] - 1) * 2))
				current_filter = self.filters[filter_i]
				for i in range(next_error.shape[1]):
					for j in range(next_error.shape[2]):
						part_E = W[i : i + current_filter.shape[0],j : j + current_filter.shape[1]]
						next_error[error_filter_i,i,j] += np.sum(part_E * np.rot90(current_filter,2)) 
						pass
					pass


		for filter_i in range(self.filters_num):
			for i in range(self.filters[filter_i].shape[0]):
				for j in range(self.filters[filter_i].shape[1]):
					W = self.input[i * self.strides[0] : i * self.strides[0] + E[filter_i].shape[0],j * self.strides[1] : j * self.strides[1] + E[filter_i].shape[1]]
					self.filters[filter_i,i,j] -= self.lr * np.sum(W * E[filter_i] * self.derivatives[self.activation](self.output[filter_i])) 
					pass

				pass

			pass

		return next_error

	def convolve2d(self,X,filter,output_dim : tuple):

		image_i = X.shape[0]
		image_j = X.shape[1]

		if self.padding == 'same':
			padding_size_i = ((output_dim[0] - 1) * self.strides[0] + self.filter_size[0]) - output_dim[0]
			padding_size_j = ((output_dim[1] - 1) * self.strides[1] + self.filter_size[1]) - output_dim[1]

			X = self.zeropad(X,(padding_size_i,padding_size_j))

			self.input = X

		convolved_image = np.zeros((output_dim[0],output_dim[1]))
		for i in range(output_dim[0]):
			for j in range(output_dim[0]):
				W = X[i * self.strides[0] : i * self.strides[0] + self.filter_size[0],j * self.strides[1] : j * self.strides[1] + self.filter_size[1]]
				convolved_image[i,j] = np.sum(W * filter)

		return convolved_image

	def zeropad(self,X,padding_size : tuple):
		output = np.zeros((X.shape[0] + padding_size[0], X.shape[1] + padding_size[1]))

		padding_size_i = int(padding_size[0] / 2 if padding_size[0] % 2 == 0 else padding_size[0] % 2)
		padding_size_j = int(padding_size[1] / 2 if padding_size[1] % 2 == 0 else padding_size[1] % 2)

		output[padding_size_i:padding_size_i + X.shape[0],padding_size_j:padding_size_j + X.shape[1]] = X

		return output

	def dilate(self,X,dilation_size : tuple):

		output = np.zeros((X.shape[0] + (X.shape[0] - 1) * dilation_size[0], X.shape[1] + (X.shape[1] - 1) * dilation_size[1]))
 
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				output[i * dilation_size[0] + i,j * dilation_size[1] + j] = X[i,j]
				pass
			pass

		return output



class pooling_layer:

	def __init__(self,type = 'max' ,pooling_size : tuple = (2,2), strides : tuple = (2,2)):
		
		self.type = type
		self.strides = strides
		self.pooling_size = pooling_size
		
		pass

	def feedfoward(self,X):
		
		self.input = X

		image_layers = X.shape[0]
		image_dim_i = X.shape[1]
		image_dim_j = X.shape[2]

		output_dim_i = int((image_dim_i - self.pooling_size[0]) / self.strides[0] + 1)
		output_dim_j = int((image_dim_j - self.pooling_size[1]) / self.strides[1] + 1)
		
		pooled_features = np.zeros((image_layers,output_dim_i,output_dim_j))
		self.where = {}
		for image_i in range(image_layers):
			for i in range(output_dim_i):
				for j in range(output_dim_j):
					W = X[image_i,i * self.strides[0] : i * self.strides[0] + self.pooling_size[0],j * self.strides[1] : j * self.strides[1] + self.pooling_size[1]]
					if self.type == 'max':
						pooled_features[image_i,i,j] = np.max(W)
						self.where[image_i,i,j] = [np.argwhere(W == np.max(W))]
					elif self.type == 'avg':
						pooled_features[image_i,i,j] = np.sum(W) / (W.shape[0] * W.shape[1])	
						pass

		self.output = pooled_features

		return self.output
		pass

	def backprop(self,E,loss,layer_i):
		unpooled_features = np.zeros((self.input.shape[0],self.input.shape[1],self.input.shape[2]))

		if self.type == 'max':
			for image_i in range(unpooled_features.shape[0]):
				for i in range(self.output.shape[1]):
					for j in range(self.output.shape[2]):
						for pos in self.where[image_i,i,j]: 
							unpooled_features[image_i,i * self.strides[0] : i * self.strides[0] + self.pooling_size[0],j * self.strides[1] : j * self.strides[1] + self.pooling_size[1]][pos[0,0],pos[0,1]] = E[image_i,i,j]
							pass
						
						pass

					pass

				pass	

			print(unpooled_features)

		elif self.type == 'avg':
			pass

		return unpooled_features

		pass

class flatten:

	def feedfoward(self,X):

		self.input = X

		flatX = X.flatten(order = 'C') 

		self.output = np.array(list(flatX),ndmin=2).T
		self.nodes = flatX.shape[0]

		return self.output

	def backprop(self,E,loss,layer_i):

		return E.reshape(self.input.shape)

		pass

	pass

class relu_layer:

	def feedfoward(self,X):
		self.input = X

		self.output = abs(X * (X > 0))

		return self.output

		pass

	def backprop(self,E,loss,layer_i):

		return E * abs(1 * (self.output > 0))

		pass

class softmax_layer:

	def feedfoward(self,X):
		self.input = X

		self.output =  np.exp(X - np.max(X)) / np.exp(X - np.max(X)).sum()

		return self.output
		pass

	def backprop(self,E,loss,layer_i):
		if layer_i == 0 and loss == 'crossentropy':
			return E
		else:
			return E * self.output * (1 - self.output) 	
		pass

#training and testing the network 

nn = NeuralNetwork()

nn.add(conv_layer(10,(2,2),padding='same'))
nn.add(relu_layer())
nn.add(conv_layer(10,(2,2),padding='same'))
nn.add(relu_layer())
nn.add(flatten())
nn.add(dense_layer(128,0.001,activation = "relu"))
nn.add(dense_layer(10,0.001,activation = "softmax"))

#loading train data and training

path_dataset = 'D:/Dev Projects/Machine Learning/NeuralNetworks/Datasets/vanilla_mnist_dataset/'

x_train = idx2numpy.convert_from_file(path_dataset + 'train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file(path_dataset + 'train-labels.idx1-ubyte')

x_train = (x_train / 255.0 * 0.99) + 0.01

nn.train(x_train,y_train,1,'crossentropy')

#loading test data and testing

x_test = idx2numpy.convert_from_file(path_dataset + 't10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file(path_dataset + 't10k-labels.idx1-ubyte')

x_test = (x_test / 255.0 * 0.99) + 0.01

scorecard = []

for t in range(len(x_test)):
	if np.argmax(nn.predict(x_test[t])) == y_test[t]:
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass

	pass

scorecard_array = np.asarray(scorecard)

print('performance is: ', scorecard_array.sum() / scorecard_array.size * 100 , '%')