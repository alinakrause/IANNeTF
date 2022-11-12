

class Layer:
	pass

	def __init__(self, n_units, input_units):
		self.n_units = n_units
		self.input_units = input_units
		
	
        	self.bias_vector = np.zeros(n_units)
        
	
		self.weight_matrix = np.random.rand(input_units, n_units)
		
		
		self.layer_input = None
		self.layer_preactivation = None
		self.layer_activation = None
		
	    def forward_step(self, input_vector):
		self.layer_input = layer_input
		self.layer_preactivation = input_vector @ self.weight_matrix + self.bias_vector
		self.layer_activation = reLu(layer_preactivation)
		return layer_activation
		

	    def backward_step(self, dLda, lr):
		    error_signal = relu_derivative(self.preactivation) * dLda
		    weights_grad = (np.transpose(self.input) @ error_signal)
		    input_grad = error_signal @ np.transpose(self.weights)
		    self.weights = self.weights - (lr * weights_grad)
		    self.bias = self.bias - (lr * error_signal)
		    return input_grad
		 


