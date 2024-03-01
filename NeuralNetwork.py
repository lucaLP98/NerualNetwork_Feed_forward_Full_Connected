"""
 In this file there are defined a class for a Feed-Forward Full Connected Neural Network
 This class contains functions for:
    - Initialize the network, defining numbers of layer, numbers of neurons for each layer, activation function for
      each layer, error function
    - Print information about the network architecture, weights and bias
    - Forward propagation through the net of an input X
    - Back propagation for gradient computation
    - Standard gradient descent for network weights update
    - RPROP algorithm for network weights update
    - learning and accuracy assessment

 File: NeuralNetwork.py
 Author: Pastore Luca N97000431
 Target: Università degli studi di Napoli Federico II
"""

import numpy as np
import copy
from LearningResults import *
from loss_functions import softmax


class NeuralNetwork:
    """
        Initialization of the Neural Network

        args:
            layer_sizes: list that contains the number of layers of the network with the respective numbers of neurons
                         (e.g. (3,5,2,1) -> indicates a net of 4 layer, first layer with 3 neurons, second with 5 neurons,...)
            loss: error function of the net with the corresponding derivative
            actv: list of activation functions for each layers with the corresponding derivatives
    """
    def __init__(self, layer_sizes, loss, actv):
        assert len(layer_sizes) == len(
            actv), "The number of levels and the number of activation functions must be the same"

        # Error function
        self.loss = loss

        # Activation functions
        self.actv = actv

        # Number of layers and neurons
        self.n_layers = len(layer_sizes)  # numbers of layers of the net
        self.layer_sizes = layer_sizes

        # Wheigts initialization
        self.weights = []  # Incoming connection weights for each neurons
        self.biases = []  # Biases for each layer
        # Generation of initial random values for weights
        for i in range(1, self.n_layers):
            self.weights.append(0.1 * np.random.randn(self.layer_sizes[i], self.layer_sizes[i - 1]))
            self.biases.append(np.zeros((1, self.layer_sizes[i])))

    """
        Printing matrices of weights of the connection of the net.
    """
    def print_weights(self):
        for i in range(self.n_layers - 1):
            print(f' Weights between layer {i} and layer {i + 1}:')
            print(self.weights[i])
            print()

    """
        Printing biases for each layers.
    """
    def print_biases(self):
        for i in range(self.n_layers - 1):
            print(f' Biases layer {i}:')
            print(self.biases[i])
            print()

    """
        Print information on the architectures of the net
    """
    def print_net_info(self):
        print("\n *** Neural Network Architecture: ***\n")
        print(" Neural Network type: Feed - Forward")
        print(" Numbers of layers: " + str(self.n_layers))
        print(" Numbers of neurons: " + str(np.sum(self.layer_sizes)))
        print(" Loss function: " + str(self.loss.__name__))
        print("\n Architecture: ")
        for i in range(self.n_layers):
            print("  - Layer " + str(i) + ": " + str(self.layer_sizes[i]) + " neurons\tActivations: " + str(self.actv[i].__name__))

    """
        Computation of output of the net with input x

        arg: 
            x -> input of the net
            sm -> flag value indicating whether or not softmax is used (sm = True use softmax, sm = False not use softmax)
        return: 
            y -> output matrix N x c (N number of sample, c number of classes)
    """
    def compute_output(self, x, sm=False):
        a, z = self.forward_propagation(x)
        y = z[-1] if not sm else softmax(z[-1])  # Computation output of the net

        return np.transpose(y)

    """
        Function that allow to compute the accuracy of the net with output y_net

        arg:
            y_net -> output, matrix N x c, with N=number of samples, c=number of class
            target -> target values for each value of x
        return:
            accuracy -> accuracy value for the net
    """
    @staticmethod
    def compute_accuracy(y_net, target):
        N = len(target)
        correct_answers = 0

        for i in range(N):
            if np.argmax(y_net[i]) == np.argmax(target[i]):
                correct_answers += 1

        accuracy = correct_answers / N

        return accuracy

    """
        Function that allow to compute the accuracy of the net with input x

        arg:
            x -> dataset, matrix N x c, with N=number of samples, c=number of features
            target -> target values for each value of x
            sm -> flag value indicating whether or not softmax is used (sm = True use softmax, sm = False not use softmax)
        return:
            accuracy -> accuracy value for the net
    """
    def net_accuracy(self, x, target, sm):
        y_net = self.compute_output(x, sm)
        accuracy = self.compute_accuracy(y_net, target)

        return accuracy

    """
        Function that allow to compute the error of the net with input x

        arg:
            x -> dataset, matrix N x c, with N=number of samples, c=number of features
            target -> target values for each value of x
            sm -> flag value indicating whether or not softmax is used (sm = True use softmax, sm = False not use softmax)
        return:
            error -> error value for the net
    """
    def compute_error(self, x, t, sm):
        y = self.compute_output(x, sm)
        error = self.loss(y, t)

        return error

    """
        Function that allow to compute the average error of the net with input x

        arg:
            x -> dataset, matrix N x c, with N=number of samples, c=number of features
            target -> target values for each value of x
            sm -> flag value indicating whether or not softmax is used (sm = True use softmax, sm = False not use softmax)
        return:
            error -> error value for the net
    """
    def compute_average_error(self, x, t, sm):
        error = self.compute_error(x, t, sm)
        average_error = error / t.shape[0]

        return average_error

    """
        Forward Propagation of input X.
        Computation of "a" values and "z" values for each neurons of the net.
            a values: input of each neuron
            z values: output of each neuron

        arg:
            x -> dataset, matrix N x c (N = Number of sample, c = Number of features)
        return 
            a -> list of matrix contains input for neurons of each layer
            z -> list of matrix contains output for neurons of each layer
    """
    def forward_propagation(self, x):
        # Correctness check of input dimensions
        assert x.shape[1] == self.layer_sizes[0], "Input size incorrect. Number of input features should match the number of neurons in the first layer."

        # Neurons's input and output initialization
        a = []  # input for each neurons
        z = []  # output of each neurons

        # Computation of "z" values for the input layer (input value "a" for the first layer is net input "x")
        a.append(np.transpose(x))
        z.append(self.actv[0](np.transpose(x)))

        # Computation of "a" values and "z" values for the hidden layers
        for i in range(len(self.weights)):
            # Input computation
            a_i = np.matmul(self.weights[i], z[- 1]) + np.transpose(self.biases[i])
            a.append(a_i)

            # Output computation
            z_i = self.actv[i + 1](a_i)
            z.append(z_i)

        return a, z

    """
        Error Back Propagation for calculation of the derivatives of the error function respect to the network weights.

        arg:
            x -> training set, matrix N x m (N = Number of sample, m = Number of features) 
            t -> target for training set elements, matrix N x c (N = Number of sample, m = Number of classes) one-hot encoded
        return:
            derivatives_weights -> derivatives of loss function respect weights for each layer (derivatives_weights[i] = derivatives for w_i - w_i-1)
            derivatives_biases -> derivatives of loss function respect biases for each layer (derivatives_weights[i] = derivatives for w_i - w_i-1)
    """
    def back_propagation(self, x, t):
        DERIVATIVE = True
        deltas = []
        derivatives_weights = []
        derivatives_biases = []

        # Forward step
        a, z = self.forward_propagation(x)

        # Computation delta value for output layer
        a_k = a[- 1]  # Input for output-layer neurons
        delta_out = self.actv[-1](a_k, DERIVATIVE) * self.loss(z[-1], np.transpose(t), DERIVATIVE)  # Delta value for output neurons
        deltas.append(delta_out)

        # Computation delta value for hidden layers
        for i in range(self.n_layers - 2, -1, -1):
            delta_i = np.transpose(self.weights[i]) @ deltas[0]
            delta_hidden = self.actv[i](a[i], DERIVATIVE) * delta_i
            deltas.insert(0, delta_hidden)

        # Computation derivatives for each layer of weights
        for layer in range(1, self.n_layers):
            derivatives_weights.append(deltas[layer] @ np.transpose(z[layer - 1]))  # Derivatives respect to weights
            derivatives_biases.append(np.sum(deltas[layer]))  # Derivatives respect to biases

        return derivatives_weights, derivatives_biases

    """
        Function that applied the standard gradient descent to update net weights

        arg:
            gradient -> gradient for the loss function respect weights at the epoch t
            bias_gradient -> gradient for the loss function respect biases at the epoch t
            eta -> hyperparameter for value update
        return: 
            weights_var_curr -> matrix contains the variation for each weights
            biases_var_curr -> matrix contains the variation for each biases
    """
    def standard_gradient_descent(self, gradient, bias_gradient, eta):
        # Initialization of matrix that will contain the delta-variation for each weight/bias
        weights_var_curr = []
        biases_var_curr = []

        for i in range(len(self.weights)):
            # Weights update
            weight_variation = -eta * gradient[i]  # Definition of variation
            weights_var_curr.append(np.array(weight_variation))  # Save the variation for the weight
            self.weights[i] = self.weights[i] + weight_variation  # Update

            # Biases update
            bias_variation = -eta * bias_gradient[i]  # Definition of variation
            biases_var_curr.append(np.array(bias_variation))  # Save the variation for the bias
            self.biases[i] = self.biases[i] + bias_variation  # Update

        return weights_var_curr, biases_var_curr

    """
        Resilient backPropagation algorithm (RPROR) standard for weights updates
        This function computate the delta value for each weights and biases of the net and use them for update weigths and biases

        arg:
            grad -> gradient for the loss function respect weights at the epoch t (array of derivatives of loss function respect to weights calculate t-th epoch)
            old_grad -> gradient for the loss function respect weights at the epoch t-1 (array of derivatives of loss function respect to weights calculate (t-1)-th epoch)
            grad_bias -> gradient for the loss function respect biases at the epoch t (array of derivatives of loss function respect to biases calculate t-th epoch)
            old_grad_bias -> gradient for the loss function respect biases at the epoch t-1 (array of derivatives of loss function respect to biases calculate (t-1)-th epoch)
            delta_w -> delta values for weights for previous era
            delta_bias -> delta values for biases for previous era
            delta_max -> max value for step-size delta
            delta_min -> minimum value for step-size delta
            eta -> hyperparameter for standard gradient descent of the first epoch
            eta_plus -> hyperparameter that allows the increase of delta. Must be eta_plus > 1
            eta_minus -> hyperparameter that allows the decrease of delta. Must be 0 < eta_minus < 1  
            epoch -> current epoch   
            weights_var_pred -> variations for each weight at epoch t-1 (epoch preceding the current one)
            biases_var_pred -> variations for each weight at epoch t-1 (epoch preceding the current one)
            rprop_method -> contain the rprop variant to use (rprop+, rprop-, irprop+, irprop-)
        return:
            weights_var_curr -> matrix contains the variation for each weights
            biases_var_curr -> matrix contains the variation for each biases
            grad -> gradient eventually updated of loss function respect weights (update is necessary for som rprop variants)
            grad_bias -> gradient eventually updated of loss function respect biases (update is necessary for som rprop variants)
    """
    def rprop(self, grad, old_grad, grad_bias, old_grad_bias, delta_w, delta_bias, delta_max, delta_min, eta, eta_plus, eta_minus, weights_var_pred, biases_var_pred, error, epoch, rprop_method):
        # Input rprop method correctness check
        valid_method = {"rprop+", "rprop-", "irprop+", "irprop-"}
        rprop_method = rprop_method.lower()
        assert rprop_method in valid_method, "Error: The selected RPROP method isn't supported"

        # Initialization of matrix that will contain the delta-variation for each weight/bias
        weights_var_curr = []
        biases_var_curr = []

        if epoch == 0:
            # For the first epoch there isn't a gradient at epoch-1. In this case the standard gradient descent is applied
            weights_var_curr, biases_var_curr = self.standard_gradient_descent(grad, grad_bias, eta)
        else:
            # For each epoch following the first rprop si applied
            for i in range(len(self.weights)):  # Iteration on layers of the net
                # Update delta value for each weight
                grad_prod = grad[i] * old_grad[i]
                delta_w[i] = np.where(grad_prod > 0, np.minimum(eta_plus * delta_w[i], delta_max), np.where(grad_prod < 0, np.maximum(eta_minus * delta_w[i], delta_min), delta_w[i]))

                # Update delta value for each bias
                grad_bias_prod = grad_bias[i] * old_grad_bias[i]
                delta_bias[i] = np.where(grad_bias_prod > 0, np.minimum(eta_plus * delta_bias[i], delta_max), np.where(grad_bias_prod < 0, np.maximum(eta_minus * delta_bias[i], delta_min), delta_bias[i]))

                # Weights variation is calculated differently based on the different version of the RPROP selected
                if rprop_method == "rprop-":  # Standard version of RPROP: RPROP without Weight-Backtracking

                    # Definition of variation for weight
                    weight_variation = -np.sign(grad[i]) * delta_w[i]
                    # Definition of variation for bias
                    bias_variation = -np.sign(grad_bias[i]) * delta_bias[i]

                elif rprop_method == "rprop+":  # RPROP with Weight-Backtracking

                    # Definition of variation for weight
                    weight_variation = np.where(grad_prod >= 0, -np.sign(grad[i]) * delta_w[i], -weights_var_pred[i])
                    # Set the derivative of loss function respect weights to 0 for future iteration
                    grad[i] = np.where(grad_prod < 0, 0, grad[i])

                    # Definition of variation for bias
                    bias_variation = np.where(grad_bias_prod >= 0, -np.sign(grad_bias[i]) * delta_bias[i], -biases_var_pred[i])
                    # Set the derivative of loss function respect biases to 0 for future iteration
                    grad_bias[i] = np.where(grad_bias_prod < 0, 0, grad_bias[i])

                elif rprop_method == "irprop-":  # Improved RPROP without Weight-Backtracking

                    # Set the derivative of loss function respect weights to 0
                    grad[i] = np.where(grad_prod < 0, 0, grad[i])
                    # Definition of variation for weight
                    weight_variation = -np.sign(grad[i]) * delta_w[i]

                    # Set the derivative of loss function respect biases to 0
                    grad_bias[i] = np.where(grad_bias_prod < 0, 0, grad_bias[i])
                    # Definition of variation for bias
                    bias_variation = -np.sign(grad_bias[i]) * delta_bias[i]

                else:  # rprop_method == "irprop+" Improved RPROP with Weight-Backtracking

                    if error[epoch] > error[epoch - 1]:
                        weights_var_gg = -weights_var_pred[i]
                        biases_var_gg = -biases_var_pred[i]
                    else:
                        weights_var_gg = 0
                        biases_var_gg = 0

                    # Definition of variation for weight
                    weight_variation = np.where(grad_prod >= 0, -np.sign(grad[i]) * delta_w[i], weights_var_gg)
                    # Definition of variation for bias
                    bias_variation = np.where(grad_bias_prod >= 0, -np.sign(grad_bias[i]) * delta_bias[i], biases_var_gg)

                    # Set the derivative of loss function respect weights to 0 for future iteration
                    grad[i] = np.where(grad_prod < 0, 0, grad[i])
                    # Set the derivative of loss function respect bias to 0 for future iteration
                    grad_bias[i] = np.where(grad_bias_prod < 0, 0, grad_bias[i])

                # Weights update
                weights_var_curr.append(np.array(weight_variation))  # Save the current variation for each weight
                self.weights[i] = self.weights[i] + weight_variation  # Update

                # Biases update
                biases_var_curr.append(np.array(bias_variation))  # Save the current variation for each bias
                self.biases[i] = self.biases[i] + bias_variation  # Update

        return weights_var_curr, biases_var_curr, grad, grad_bias, delta_w, delta_bias

    """
        Learning phase with batch update and RPROP algorithms.
        The algorithm, for each epochs:
         - compute the step size with rprop and use it for update weights and biases
         - evaluate the error on training set and validation set
         - evaluate the accuracy on training set and validation set
         - minimizes the error on the validation set and save the best configuration of weights and biases

        arg:
            x_train -> training set
            t_train -> targets for training set elements
            x_val -> validation set
            t_val -> targets for validation set elements
            delta_init -> initial delta value
            delta_max -> max value for step-size delta
            delta_min -> minimum value for step-size delta
            eta -> hyperparameter for standard gradient descent of the first epoch
            eta_plus -> hyperparameter that allows the increase of delta. Must be eta_plus > 1
            eta_minus -> hyperparameter that allows the decrease of delta. Must be 0 < eta_minus < 1      
            max_epochs -> number of epochs needed to examine the dataset
            rprop_method -> contain the rprop variant to use (rprop+, rprop-, irprop+, irprop-)
            sm -> flag value indicating whether or not softmax is used (sm = True use softmax, sm = False not use softmax)
        return:
            result -> class containing result of learning phase:
             - min_weights -> best weights configuration (min error on validation set)
             - min_biases  -> best biases configuration (min error on validation set)
             - min_epoch -> epoch with minor error on validation set
             - training_accuracies -> array with accuracy computated on training set for each epoch
             - validation_accuracies  -> array with accuracy computated on validation set for each epoch 
             - training_errors -> array with error computated on training set for each epoch
             - validation_errors -> array with error computated on validation set for each epoch
    """
    def learning(self, x_train, t_train, x_val, t_val, delta_init, delta_max, delta_min, eta, eta_plus, eta_minus, max_epochs, rprop_method, sm=False):
        # Input rprop method correctness check
        valid_method = {"rprop+", "rprop-", "irprop+", "irprop-"}
        rprop_method = rprop_method.lower()
        assert rprop_method in valid_method, "Error: The selected RPROP method is not supported"

        old_grad = []  # For each epoch t, it will contain the gradient of loss function respect weights at epoch t-1
        old_grad_bias = []  # For each epoch t, it will contain the gradient of loss function respect biases at epoch t-1

        weights_var = [np.zeros_like(weight, dtype=float) for weight in self.weights]  # Will contain the variation for each weight after rprop application
        biases_var = [np.zeros_like(weight, dtype=float) for weight in self.weights]  # Will contain the variation for each bias after rprop application

        training_errors = np.zeros(max_epochs, dtype=float)  # Array for error on training set for each epoch
        validation_errors = np.zeros(max_epochs, dtype=float)  # Array for error on validation set for each epoch
        training_accuracies = np.zeros(max_epochs, dtype=float)  # Array for accuracy on training set for each epoch
        validation_accuracies = np.zeros(max_epochs, dtype=float)  # Array for accuracy on validation set for each epoch

        # Initialization of delta_w and delta_bias with value delta_init
        delta_w = [np.full_like(weight, delta_init) for weight in self.weights]
        delta_bias = [np.full_like(bias, delta_init) for bias in self.biases]

        # Initialization of values for minimum error on validation set
        min_error = self.compute_error(x_val, t_val, sm)  # Initial error on validation set
        min_weights = copy.deepcopy(self.weights)  # Initial weights configuration
        min_biases = copy.deepcopy(self.biases)  # Initial biases configuration
        min_epoch = 0

        for epoch in range(max_epochs):
            print(" - Epoch = " + str(epoch))

            # Computation of gradient of loss function (respect weights/biases)
            grad, grad_bias = self.back_propagation(x_train, t_train)

            # Apply rprop algorithm and return the weight/bias variations
            # rprop function provide to update the weights of the network
            weights_var, biases_var, old_grad, old_grad_bias, delta_w, delta_bias = \
                self.rprop(grad, old_grad, grad_bias, old_grad_bias, delta_w, delta_bias, delta_max, delta_min, eta, eta_plus, eta_minus, weights_var, biases_var, validation_errors, epoch, rprop_method)

            y_train = self.compute_output(x_train, sm)
            y_val = self.compute_output(x_val, sm)

            # Computation of error on training-set
            training_error = self.loss(y_train, t_train)
            training_errors[epoch] = training_error
            # Computation of accuracy on training-set
            training_accuracy = self.compute_accuracy(y_train, t_train)
            training_accuracies[epoch] = training_accuracy

            # Computation of error on validation-set
            validation_error = self.loss(y_val, t_val)
            validation_errors[epoch] = validation_error
            # Computation of accuracy on validation-set
            validation_accuracy = self.compute_accuracy(y_val, t_val)
            validation_accuracies[epoch] = validation_accuracy

            # Check best net parameters
            if validation_error < min_error:
                min_error = validation_error  # Update minimum error with current validation error
                min_weights = copy.deepcopy(self.weights)  # Save the best combination for weights
                min_biases = copy.deepcopy(self.biases)  # Save the best combination for biases
                min_epoch = epoch  # Save the current epoch

        # Set the network values that minimize the error on the validation-set
        self.weights = copy.deepcopy(min_weights)  # Set the best weights for the net
        self.biases = copy.deepcopy(min_biases)  # Set the best biases for the net

        result = LearningResults(min_weights, min_biases, min_epoch, max_epochs, training_accuracies, validation_accuracies,
                                 training_errors, validation_errors, x_train.shape[0], x_val.shape[0],
                                 delta_init, delta_max, delta_min, eta_plus, eta_minus, rprop_method)

        return result


"""
 End File NeuralNetwork.py
"""