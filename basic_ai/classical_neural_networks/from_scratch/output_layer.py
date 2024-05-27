import numpy as np 
from hidden_layer import Layer

# TODO: Implement other Output layers with other types of loss functions.  

class Output_layer(Layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def define_activation_function(self, activation_function:str):
        """
        Defines the activation function and its derivative based on the provided activation function name.

        Parameters:
        ----------
        activation_function : str
            The name of the activation function to use in the output layer (e.g., 'relu', 'tanh', 'leaky_relu', 'elu', 'sigmoid', 'softmax').
        """
        if activation_function == 'relu':
            self.activation_function = lambda x: np.maximum(0, x)
            self.activation_derivative = lambda x: np.where(x > 0, 1, 0)
        
        elif activation_function == 'tanh':
            self.activation_function = lambda x: np.tanh(x)
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2       

        elif activation_function == 'leaky_relu':
            self.activation_function = lambda x: np.maximum(self.alpha*x, x)
            self.activation_derivative = lambda x: np.where(x > 0, 1, self.alpha)

        elif activation_function == 'elu':
            self.activation_function = lambda x: np.where(x > 0, x, self.alpha*(np.exp(x) - 1))
            self.activation_derivative = lambda x: np.where(x > 0, 1, self.alpha*np.exp(x))
        
        elif activation_function == 'sigmoid':
            self.activation_function = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: x * (1 - x)

        elif activation_function == 'softmax':
            self.activation_function = lambda x: np.exp(x) / np.sum(np.exp(self.node_value[0]))
            self.activation_derivative = lambda x: x * (1 - x)

class Output_layer_MSE(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)
    
    def calculate_error_term(self, target):
        '''Calculates the error term for the output layer using the Mean Squared Error (MSE) loss function.'''
        self.error_term = (target - self.node_output) / self.n_neurons
        for i in range(self.n_neurons):
            self.error_term[0][i] *= self.activation_derivative(self.node_value[0][i])

class Output_layer_MAE(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass

class Output_layer_Log_Cosh(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass

class Output_layer_Bin_CrossEntropy(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass

class Output_layer_Cat_CrossEntropy(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass

class Output_layer_SparseCat_CrossEntropy(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass

class Output_layer_Poisson(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass

class Output_layer_Cosine_Proximity(Output_layer):
    def __innit__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str):
        super().__init__(n_neurons, nprev_neurons, learning_rate, activation_function)

    def calculate_error_term(self, target):
        pass