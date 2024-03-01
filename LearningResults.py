"""
 In this file there are defined a class for contain the result of learning phase

 File: LearningResults.py
 Author: Pastore Luca N97000431
 Target: Università degli studi di Napoli Federico II
"""


from matplotlib import pyplot as plt

class LearningResults:
    """
        Initialization of the LearningResult Object

        args:
            min_weights -> best combination for net weights
            min_biases -> best combination for net biases
            min_epoch -> epoch with minimum error on validation set
            max_epoch -> total number of epoch in learning phase
            train_accuracy -> array contain accuracies on training set for each epoch
            val_accuracy -> array contain accuracies on validation set for each epoch
            train_error -> array contain errors on training set for each epoch
            val_error -> array contain errors on validation set for each epoch
            num_sample_train -> number of sample for training set
            num_sample_val -> number of sample for validation set
            delta_init -> initial delta value
            delta_max -> max value for step-size delta
            delta_min -> minimum value for step-size delta
            eta -> hyperparameter for standard gradient descent of the first epoch
            eta_plus -> hyperparameter that allows the increase of delta. Must be eta_plus > 1
            eta_minus -> hyperparameter that allows the decrease of delta. Must be 0 < eta_minus < 1
    """
    def __init__(self, min_weights, min_biases, min_epoch, max_epoch, train_accuracy, val_accuracy, train_error, val_error, num_sample_train, num_sample_val,
                 delta_init, delta_max, delta_min, eta_plus, eta_min, rprop_method):
        # Learning Results
        self.min_weights = min_weights
        self.min_biases = min_biases
        self.min_epoch = min_epoch
        self.training_accuracy = train_accuracy
        self.validation_accuracy = val_accuracy
        self.training_error = train_error
        self.validation_error = val_error
        self.num_sample_train = num_sample_train
        self.num_sample_val = num_sample_val

        # Learning parameters
        self.max_epoch = max_epoch
        self.delta_init = delta_init
        self.delta_max = delta_max
        self.delta_min = delta_min
        self.eta_plus = eta_plus
        self.eta_minus = eta_min
        self.rprop_method = rprop_method

    """
        Print parameter used for rprop algorithms
    """
    def print_learning_parameter(self):
        print("\n\n ********** LEARNING PARAMETER **********")
        print(" Total epochs : " + str(self.max_epoch))
        print(" RPROP Parameter:")
        print(" - RPROP method: " + str(self.rprop_method))
        print(" - Initial delta value: " + str(self.delta_init))
        print(" - Max delta: " + str(self.delta_max))
        print(" - Min delta: " + str(self.delta_min))
        print(" - eta plus: " + str(self.eta_plus))
        print(" - eta minus: " + str(self.eta_minus))

    """
        Print learning result on training set
    """
    def print_trining_result(self):
        print("\n\n ********** RESULTS ON TRAINING SET **********")
        print(" Epoch with minor error on validation set = " + str(self.min_epoch) + "\n")
        print(" Average Training Error = " + str(self.training_error[self.min_epoch] / self.num_sample_train))
        print(" Training Accuarcy = " + str(self.training_accuracy[self.min_epoch] * 100) + "%\n")

    """
        Print learning result on validation set
    """
    def print_validation_result(self):
        print("\n\n ********** RESULTS ON VALIDATION SET **********")
        print(" Epoch with minor error on validation set = " + str(self.min_epoch) + "\n")
        print(" Average Validation Error = " + str(self.validation_error[self.min_epoch] / self.num_sample_val))
        print(" Validation Accuarcy = " + str(self.validation_accuracy * 100) + "%")

    """
        Print learning result on training and validation set
    """
    def print_result(self):
        print("\n\n ********** LEARNING RESULTS **********")
        print(" Epoch with minor error on validation set = " + str(self.min_epoch) + "\n")
        print(" Average Training Error = " + str(self.training_error[self.min_epoch] / self.num_sample_train))
        print(" Training Accuarcy = " + str(self.training_accuracy[self.min_epoch] * 100) + "%\n")
        print(" Average Validation Error = " + str(self.validation_error[self.min_epoch] / self.num_sample_val))
        print(" Validation Accuarcy = " + str(self.validation_accuracy[self.min_epoch] * 100) + "%")

    """
        Print array containing error on training set for each epoch
    """
    def print_training_error_array(self):
        print("\n\n Error Training set for " + str(self.max_epoch) + " epochs")
        print(self.training_error)

    """
        Print array containing error on validation set for each epoch
    """
    def print_validation_error_array(self):
        print("\n\n Error Validation set " + str(self.max_epoch) + " epochs")
        print(self.validation_error)

    """
        Print array containing accuracy on training set for each epoch
    """
    def print_training_accuracy_array(self):
        print("\n\n Accuracy on Training set for " + str(self.max_epoch) + " epochs")
        print(self.training_accuracy)

    """
        Print array containing accuracy on validation set for each epoch
    """
    def print_validation_accuracy_array(self):
        print("\n\n Accuracy on Validation set for " + str(self.max_epoch) + " epochs")
        print(self.validation_accuracy)

    """
        Print graph for accuracy on validation set
    """
    def print_accuracy_graph_validation(self):
        plt.figure(figsize=(14, 10))
        plt.plot(self.validation_accuracy)
        plt.title('Accuracy Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    """
        Print graph for accuracy on training set
    """
    def print_accuracy_graph_training(self):
        plt.figure(figsize=(14, 10))
        plt.plot(self.training_accuracy)
        plt.title('Accuracy Training Set')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.show()

    """
        Print graph for error on validation set
    """
    def print_error_graph_validation(self):
        plt.figure(figsize=(14, 10))
        plt.plot(self.validation_error)
        plt.title('Error Validation Set')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

    """
        Print graph for error on training set
    """
    def print_error_graph_training(self):
        plt.figure(figsize=(14, 10))
        plt.plot(self.training_error)
        plt.title('Error Training Set')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

    """
        Print graph with comparison between training error and validation error
    """
    def print_error_graph(self):
        plt.figure(figsize=(14, 10))
        plt.plot(self.training_error, label="Training-Set", color='blue')
        plt.plot(self.validation_error, label="Validation-Set", color='red')
        plt.title('Learning Error')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        plt.show()

    """
        Print graph with comparison between training accuracy and validation accuracy
    """
    def print_accuracy_graph(self):
        plt.figure(figsize=(14, 10))
        plt.plot(self.training_accuracy, label="Training-Set", color='blue')
        plt.plot(self.validation_accuracy, label="Validation-Set", color='red')
        plt.title('Learning Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.show()


"""
    End file LearningResults.py
"""