import numpy as np

'''So when constructing a Layer im obligated to make some choices, 
such as, how to initialize the weights/bias, how to use the properties for each 
different type of neural network, and others, so I'll let math guide me along with
some code preferences aswell. All of them will be registred in the mathematical
guide of this repo, this text is a reminder for myself not forget to register that =)'''

# So lets assume the notation that every node is self suficient, it contains its own bias,
# and weights for own calculations.

# TODO: search the impact on how to implement optimzier inside neural networks like Adam, RMSprop, etc.

# TODO: search the impact on how to implement regularization techniques like L1, L2, Dropout, etc.

class Layer:
    '''A class representing a single layer in a neural network.

    Attributes:
    ----------
    n_neurons : int
        The number of neurons in the layer.
    nprev_neurons : int
        The number of neurons in the previous layer.
    learning_rate : float
        The learning rate for weight updates during backpropagation.
    weights : np.ndarray
        The weight matrix of shape (nprev_neurons, n_neurons).
    bias : np.ndarray
        The bias vector of shape (1, n_neurons).
    activation_function : callable
        The activation function used in the layer.
    activation_derivative : callable
        The derivative of the activation function used in the layer.
    node_value : np.ndarray or None
        The value of the nodes before activation.
    node_output : np.ndarray or None
        The output of the nodes after activation.
    alpha : float
        The alpha parameter for Leaky ReLU and ELU activation functions, default is 0.1.
    error_term : np.ndarray or None
        The error term for the layer used in backpropagation.
    weights_init : str
        The method used to initialize the weights, default is 'random'.
    bias_init : str
        The method used to initialize the bias, default is 'zeros'.
    '''
    

    def __init__(self, n_neurons:int, nprev_neurons:int, learning_rate:float, activation_function:str, alfa = 0.1, weights_init = 'random', bias_init = 'zeros') -> None:
        self.n_neurons = n_neurons
        self.nprev_neurons = nprev_neurons
        self.learning_rate = learning_rate
        self.weight_initialization(weights_init)
        self.bias_initialization(bias_init)
        self.define_activation_function(activation_function)
        self.node_value = None
        self.node_output = None

        if(activation_function == 'leaky_relu' or activation_function == 'elu'):
            assert alfa > 0, 'The alpha parameter must be greater than 0 for Leaky ReLU and ELU activation functions.'
            assert type(alfa) == float, 'The alpha parameter must be a float.'

            self.alpha = alfa

        
    def weight_initialization(self, weights_init:str):
        ''' ''' 
        if(weights_init == 'he'):
            self.weights = np.random.randn(self.nprev_neurons, self.n_neurons) * np.sqrt(2. / self.nprev_neurons)
        elif(weights_init == 'random'):
            self.weights =  np.random.randn(self.nprev_neurons, self.n_neurons) 
        elif(weights_init == 'zeros'):
            self.weights = np.zeros(self.nprev_neurons, self.n_neurons)
        elif(weights_init == 'ones'):
            self.weights = np.ones(self.nprev_neurons, self.n_neurons)
        elif weights_init == 'xavier' or weights_init == 'glorot':
            self.weights = np.random.randn(self.nprev_neurons, self.n_neurons) * np.sqrt(1. / self.nprev_neurons)
        else: 
            raise ValueError('Invalid weight initialization method.')


    def bias_initialization(self, bias_init:str):
        ''' '''
        if(bias_init == 'zeros'):
            self.bias = np.zeros(1, self.n_neurons)
        elif(bias_init == 'ones'):
            self.bias = np.ones(1, self.n_neurons)
        elif(bias_init == 'random'):
            self.bias = np.random.randn(1, self.n_neurons) * np.sqrt(2. / self.nprev_neurons)
        else:
            raise ValueError('Invalid bias initialization method.')

    def define_activation_function(self, activation_function:str):
        """
        Defines the activation function and its derivative based on the provided activation function name.

        Parameters:
        ----------
        activation_function : str
            The name of the activation function to use (e.g., 'relu', 'tanh', 'leaky_relu', 'elu', 'sigmoid').
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

        else:
            raise ValueError('Invalid activation function name.')
        
        
    def forward_pass(self, prev_out_layer):
        """
        Computes the forward pass by applying the activation function to the weighted sum of inputs plus bias.

        Parameters:
        ----------
        prev_out_layer : np.ndarray
            The output from the previous layer.
        """

        self.node_value = np.dot(prev_out_layer, self.weights) + self.bias
        self.node_output = np.array([self.activation_function(node) for node in self.node_value])

    def calculate_error_term(self, next_error_term, next_weights):
        """
        Calculates the error term for the current layer using the error term and weights of the next layer.

        Parameters:
        ----------
        next_error_term : np.ndarray
            The error term from the next layer.
        next_weights : np.ndarray
            The weights from the next layer.
        """

        self.error_term = np.dot(next_error_term, next_weights.T)
        for i in range(self.n_neurons):
            self.error_term[0][i] *= self.activation_derivative(self.node_value[0][i])

    def backpropagation_update(self, prev_out_layer):
        """
        Updates the weights and biases of the layer using the calculated error term and the learning rate.

        Parameters:
        ----------
        prev_out_layer : np.ndarray
            The output from the previous layer.
        """

        self.weights = self.weights - self.learning_rate * (np.dot(prev_out_layer.T,self.error_term))
        self.bias = self.bias - self.learning_rate * self.error_term