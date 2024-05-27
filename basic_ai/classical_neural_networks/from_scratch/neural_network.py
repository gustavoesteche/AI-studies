import numpy as np
from hidden_layer import Layer
from output_layer import Output_layer_MSE

# TODO: Implement other types of neural networks

class Neural_network:
    def __init__(self, layer_list) -> None:
        self.layers = layer_list
        self.n_layers = len(layer_list)        

    def forward_pass(self, input_data):
        prev_out_layer = input_data
        for layer in self.layers:
            layer.forward_pass(prev_out_layer)
            prev_out_layer = layer.node_output

    def backpropagation(self, input_data, target):
        self.layers[-1].calculate_error_term(target)
        next_error_term = self.layers[-1].error_term
        next_weights = self.layers[-1].weights

        for i in range(self.n_layers-2, -1, -1):
            self.layers[i].calculate_error_term(next_error_term, next_weights)
            next_error_term = self.layers[i].error_term
            next_weights = self.layers[i].weights

        for i in range(self.n_layers-1, 0, -1):
            self.layers[i].backpropagation_update(self.layers[i - 1].node_output)

        self.layers[0].backpropagation_update(np.array([input_data]))

    def train(self, data, target, epochs:int):
        for epoch in range(epochs):
            for i in range(len(data)):
                self.forward_pass(data[i])
                self.backpropagation(data[i], target[i])

    def predict(self, data):
        self.forward_pass(data)
        return self.layers[-1].node_output
    
    def accuracy(self, data_test, target, round=True):
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

