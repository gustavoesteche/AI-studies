import numpy as np

'''So when constructing a Layer im obligated to make some choices, 
such as, how to initialize the weights/bias, how to use the properties for each 
different type of neural network, and others, so I'll let math guide me along with
some code preferences aswell. All of them will be registred in the mathematical
guide of this repo, this text is a reminder for myself not forget to register that =)'''

# So lets assume the notation that every node is self suficient, it contains its own bias,
# and weights for own calculations.

class Layer:
    def __init__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str) -> None:
        self.n_neurons = n_neurons
        self.nprev_neurons = nprev_neurons
        self.learning_rate = learning_rate
        self.weights = np.random.rand(nprev_neurons, n_neurons)
        self.bias = np.random.rand(n_neurons)
        self.define_activation_function(activation_function)
        self.node_value = None
        self.node_output = None
        
    def define_activation_function(self, activation_function:str):
        if activation_function == 'relu':
            self.activation_function = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: np.where(x > 0, 1, 0)
        
        elif activation_function == 'sigmoid':
            self.activation_function = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: x * (1 - x)
        

    def forward_pass(self, prev_out_layer):
        self.node_value = np.dot(prev_out_layer, self.weights) + self.bias
        self.node_output = np.array([self.activation_function(node) for node in self.node_value])

    def calculate_error_term(self, next_error_term, next_weights):
        self.error_term = np.dot(next_error_term, next_weights.T)
        for i in range(self.n_neurons):
            self.error_term[i] *= self.activation_derivative(self.node_value[i])

    def backpropagation_update(self, prev_out_layer):
        for i in range(self.nprev_neurons):
            self.weights[i] = self.weights - self.learning_rate * prev_out_layer[i] * self.error_term

        self.bias = self.bias - self.learning_rate * self.error_term


class Output_layer(Layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)
    
    def calculate_error_term(self, target):
        self.error_term = target - self.node_output
        for i in range(self.n_neurons):
            self.error_term[i] *= self.activation_derivative(self.node_value[i])
