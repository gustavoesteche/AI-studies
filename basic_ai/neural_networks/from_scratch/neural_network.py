import numpy as np

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
            self.activation_derivative = lambda x: self.activation_function(x) * (1 - self.activation_function(x))
        

    def forward_pass(self, prev_out_layer):
        self.node_value = np.dot(prev_out_layer, self.weights) + self.bias
        self.node_output = np.array([self.activation_function(node) for node in self.node_value])

    def calculate_error_term(self, next_error_term, next_weights):
        self.error_term = np.dot(next_error_term, next_weights.T)
        for i in range(self.n_neurons):
            self.error_term[i] *= self.activation_derivative(self.node_value[i])

    def backpropagation_update(self, prev_out_layer):
        for i in range(self.nprev_neurons):
            self.weights[i] = self.weights[i] - self.learning_rate * prev_out_layer[i] * self.error_term

        self.bias = self.bias - self.learning_rate * self.error_term


class Output_layer(Layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)
    
    def calculate_error_term(self, target):
        self.error_term = target - self.node_output
        for i in range(self.n_neurons):
            self.error_term[i] *= self.activation_derivative(self.node_value[i])

class Neural_network:
    def __init__(self, n_layers:int, nodes_per_layer:int, learning_rate:float, output_nodes:int, input_size:int, hidden_activation_function = 'relu', output_function='relu') -> None:
        self.output_nodes = output_nodes
        self.input_size = input_size
        self.learning_rate = learning_rate	
        self.n_p_layer = nodes_per_layer
        self.n_layers = n_layers
        self.hidden_activation_function = hidden_activation_function
        self.output_function = output_function
        self.layers = []
        self.construct_layers()

    def construct_layers(self):
        self.layers.append(Layer(self.n_p_layer, self.input_size,self.learning_rate, self.hidden_activation_function))
        
        for _ in range(self.n_layers-1):
                self.layers.append(Layer(self.n_p_layer, self.n_p_layer, self.learning_rate, self.hidden_activation_function))
                
        self.layers.append(Output_layer(self.output_nodes, self.n_p_layer, self.learning_rate, self.output_function))        
    
    def forward_pass(self, input_data):
        prev_out_layer = input_data
        for layer in self.layers:
            layer.forward_pass(prev_out_layer)
            prev_out_layer = layer.node_output

    def backpropagation(self, input_data, target):
        self.layers[-1].calculate_error_term(target)
        self.layers[-1].backpropagation_update(self.layers[-2].node_output)
        next_error_term = self.layers[-1].error_term
        next_weights = self.layers[-1].weights

        for i in range(self.n_layers-1, -1, -1):
            self.layers[i].calculate_error_term(next_error_term, next_weights)
            next_error_term = self.layers[i].error_term
            next_weights = self.layers[i].weights

        for i in range(self.n_layers, 0, -1):
            self.layers[i].backpropagation_update(self.layers[i - 1].node_output)
        self.layers[0].backpropagation_update(input_data)


    def train(self, data, target, epochs:int):
        for epoch in range(epochs):
            for i in range(len(data)):
                self.forward_pass(data[i])
                self.backpropagation(data[i], target[i])

    def predict(self, data):
        self.forward_pass(data)
        return self.layers[-1].node_output
    
    def accuracy(self):
        pass

    def mean_square_error(self, data_test, target):
        results = [self.predict(data_test[i]) for i in range(len(data_test))]
        return np.mean((results- target) ** 2)

    def print_layer_contet(self):
        for layer in self.layers:
            print(layer.node_output)
            print('-------------------')

    def print_layer_weights(self):
        for layer in self.layers:
            print(layer.weights)
            print("----------------------")
