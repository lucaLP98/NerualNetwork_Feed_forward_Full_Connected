"""
 In this file there are a main function for test the network

 File: main.py
 Author: Pastore Luca N97000431
 Target: Università degli studi di Napoli Federico II
"""

from NeuralNetwork import *
from activation_functions import *
from loss_functions import *
from dataset import *


"""
    Function used to download MNIST data and manipulate it to make it acceptable to the network
    
    return:
        train_m -> Training Set
        train_label_m -> Labels for Training set elements
        valid_m -> Validation Set
        valid_label_m -> Labels for Validation set elements
        test_m -> Test Set
        test_label_m -> Labels for Test Set elements
"""
def prepare_MNIST_data():
    (train_X, train_y), (test_X, test_y) = download_data()
    (train, train_label), (valid, valid_label) = split_training_set(train_X, train_y)

    train_m, train_label_m = manipule_data(train, train_label)
    valid_m, valid_label_m = manipule_data(valid, valid_label)
    test_m, test_label_m = manipule_data(test_X, test_y)

    return train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m


"""
    Function that allow to execute n_test for the selected rprop method and returns the average over n_tests
    for each test, the net is initialized
    
    arg:
        rprop_method -> selected rprop method
        epoch -> number of epoch for learning phase
        n_test -> number of test to execute
        eta_p -> value for eta_plus
        eta_m -> value for eta_minus
        n_neurons -> number of neurons for hidden layer
        train_m -> Training Set
        train_label_m -> Labels for Training set elements
        valid_m -> Validation Set
        valid_label_m -> Labels for Validation set elements
        test_m -> Test Set
        test_label_m -> Labels for Test Set elements
    return:
        average_error_validation_arr -> array with the average validation error for each epoch
        average_accuracy_validation_arr -> array with the average validation accuracy for each epoch
        average_error_testset -> average error on Test-Set
        standard_deviation_error_testset -> standard deviation for average error on Test-Set
        average_accuracy_testset -> average accuracy on Test-Set
        standard_deviation_accuracy_testset -> standard deviation for average accuracy on Test-Set
        average_convergence -> average convergence epoch
        standard_deviation_convergence -> standard deviation for average convergence epoch
        average_error_training_arr -> array with the average training error for each epoch
        average_accuracy_training_arr -> array with the average training accuracy for each epoch
"""
def test_rprop(rprop_method, epoch, n_test, eta_p, eta_m, n_neurons, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m):
    delta_initial = 0.0125
    delta_max = 50
    delta_min = 0
    eta = 0.0005
    eta_plus = eta_p
    eta_minus = eta_m
    num_epochs = epoch
    num_test = n_test
    hidden_neurons_num = n_neurons
    method = rprop_method
    sm = True

    input_neurons = train_m.shape[1]
    net_layers_arc = [input_neurons, hidden_neurons_num, 10]
    net_act_fun = [identity, sigmoid, identity]

    error_training = []
    accuracy_training = []

    error_validation = []
    accuracy_validation = []

    test_set_errors = []
    test_set_accuracy = []

    convergence = []

    for i in range(num_test):
        print(" TEST " + rprop_method + " number = " + str(i))
        net = NeuralNetwork(net_layers_arc, cross_entropy_sm, net_act_fun)
        learn_result = net.learning(train_m, train_label_m, valid_m, valid_label_m, delta_initial, delta_max, delta_min, eta, eta_plus, eta_minus, num_epochs, method, sm)

        error_training.append(learn_result.training_error)
        accuracy_training.append(learn_result.training_accuracy)

        error_validation.append(learn_result.validation_error)
        accuracy_validation.append(learn_result.validation_accuracy)

        test_set_errors.append(net.compute_average_error(test_m, test_label_m, sm))
        test_set_accuracy.append(net.net_accuracy(test_m, test_label_m, sm))

        convergence.append(learn_result.min_epoch)

    error_training = np.array(error_training)
    accuracy_training = np.array(accuracy_training)

    error_validation = np.array(error_validation)
    accuracy_validation = np.array(accuracy_validation)

    test_set_errors = np.array(test_set_errors)
    test_set_accuracy = np.array(test_set_accuracy)

    convergence = np.array(convergence)

    # Average error computation on validation set for each epoch
    average_error_validation_arr = np.mean(error_validation, axis=0)
    # Average accuracy computation on validation set for each epoch
    average_accuracy_validation_arr = np.mean(accuracy_validation, axis=0)

    # Average error computation on training set for each epoch
    average_error_training_arr = np.mean(error_training, axis=0)
    # Average accuracy computation on training set for each epoch
    average_accuracy_training_arr = np.mean(accuracy_training, axis=0)

    # Calculation of average error on test sets with standard deviation
    average_error_testset = np.mean(test_set_errors)
    standard_deviation_error_testset = np.std(test_set_errors)

    # Average accuracy calculation on test sets with standard deviation
    average_accuracy_testset = np.mean(test_set_accuracy)
    standard_deviation_accuracy_testset = np.std(test_set_accuracy)

    # Calculation of average convergence with standard deviation
    average_convergence = np.mean(convergence)
    standard_deviation_convergence = np.std(convergence)

    return average_error_validation_arr, average_accuracy_validation_arr, \
           average_error_testset, standard_deviation_error_testset, \
           average_accuracy_testset, standard_deviation_accuracy_testset, \
           average_convergence, standard_deviation_convergence, \
           average_error_training_arr, average_accuracy_training_arr


"""
    Function that allow to print a graph using mathplot
    
    arg: 
        *arrays -> list of array (function) to print
        names -> list of labels for each functions
        title -> graph name
        xlabel -> name for X axis
        ylabel -> name for Y axis
"""
def print_graph(*arrays, names, title, xlabel='X axis', ylabel='Y axis'):
    if len(arrays) == 0:
        print("No array to print")
        return

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure(figsize=(18, 14))

    for i, array in enumerate(arrays):
        color = colors[i % len(colors)]
        name = names[i]
        plt.plot(array, label=name, color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()


"""
    Function that allow to print result of learning on a specific dataset
    
    arg:
        rprop_method -> string, selected rprop algorithm
        average_error -> average error on selected dataset
        standard_deviation_error -> standard deviation for average error on selected dataset
        average_accuracy -> average accuracy on selected dataset
        standard_deviation_accuracy -> standard deviation for average accuracy on selected dataset
        average_convergence -> average convergence epoch
        standard_deviation_convergence -> standard deviation for average convergence epoch
        set_name -> name of selected dataset
"""
def print_result(rprop_method, average_error, standard_deviation_error, average_accuracy, standard_deviation_accuracy, average_convergence, standard_deviation_convergence, set_name):
    print("\n " + rprop_method +"\n * Error " + set_name + " = "
          + str(average_error) + "+-" + str(standard_deviation_error)
          + "\n * Accuracy " + set_name + " = " + str(average_accuracy) + "+-" + str(standard_deviation_accuracy)
          + "\n * Convergence = " + str(average_convergence) + "+-" + str(standard_deviation_convergence))


"""
    Function that allows you to evaluate the four different versions of rprop
    
    arg:
        epoch -> numbero of epoch for learning phase
        n_test -> number of test for each rprop variant
        eta_p -> value for eta_plus
        eta_m -> value for eta_minus
        n_neurons -> number of neurons for hidden layer
"""
def evaluation_rprop_version(epoch, n_test, eta_p, eta_m, n_neurons):
    train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m = prepare_MNIST_data()

    if isinstance(eta_p, list):  # Compare different rprop variant - for each variant using a specific model
        # rprop- iper-parameter
        eta_p_rprop_min = eta_p[0]
        eta_m_rprop_min = eta_m[0]
        n_neurons_rprop_min = n_neurons[0]

        # rprop+ iper-parameter
        eta_p_rprop_plus = eta_p[1]
        eta_m_rprop_plus = eta_m[1]
        n_neurons_rprop_plus = n_neurons[1]

        # irprop- iper-parameter
        eta_p_irprop_min = eta_p[2]
        eta_m_irprop_min = eta_m[2]
        n_neurons_irprop_min = n_neurons[2]

        # irprop+ iper-parameter
        eta_p_irprop_plus = eta_p[3]
        eta_m_irprop_plus = eta_m[3]
        n_neurons_irprop_plus = n_neurons[3]
    else:  # Compare different rprop variant - using the same model for each one
        # rprop- iper-parameter
        eta_p_rprop_min = eta_p
        eta_m_rprop_min = eta_m
        n_neurons_rprop_min = n_neurons

        # rprop+ iper-parameter
        eta_p_rprop_plus = eta_p
        eta_m_rprop_plus = eta_m
        n_neurons_rprop_plus = n_neurons

        # irprop- iper-parameter
        eta_p_irprop_min = eta_p
        eta_m_irprop_min = eta_m
        n_neurons_irprop_min = n_neurons

        # irprop+ iper-parameter
        eta_p_irprop_plus = eta_p
        eta_m_irprop_plus = eta_m
        n_neurons_irprop_plus = n_neurons

    # RPROP-
    rprop1_average_error_validation_arr, rprop1_average_accuracy_validation_arr, rprop1_average_error_testset, \
    rprop1_standard_deviation_error_testset, rprop1_average_accuracy_testset, rprop1_standard_deviation_accuracy_testset, \
    rprop1_average_convergence, rprop1_standard_deviation_convergence, rprop1_average_error_training_arr, rprop1_average_accuracy_training_arr \
        = test_rprop("rprop-", epoch, n_test, eta_p_rprop_min, eta_m_rprop_min, n_neurons_rprop_min, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m)

    # RPROP+
    rprop2_average_error_validation_arr, rprop2_average_accuracy_validation_arr, rprop2_average_error_testset, \
    rprop2_standard_deviation_error_testset, rprop2_average_accuracy_testset, rprop2_standard_deviation_accuracy_testset, \
    rprop2_average_convergence, rprop2_standard_deviation_convergence, rprop2_average_error_training_arr, rprop2_average_accuracy_training_arr \
        = test_rprop("rprop+", epoch, n_test, eta_p_rprop_plus, eta_m_rprop_plus, n_neurons_rprop_plus, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m)

    # iRPROP-
    rprop3_average_error_validation_arr, rprop3_average_accuracy_validation_arr, rprop3_average_error_testset, \
    rprop3_standard_deviation_error_testset, rprop3_average_accuracy_testset, rprop3_standard_deviation_accuracy_testset, \
    rprop3_average_convergence, rprop3_standard_deviation_convergence, rprop3_average_error_training_arr, rprop3_average_accuracy_training_arr \
        = test_rprop("irprop-", epoch, n_test, eta_p_irprop_min, eta_m_irprop_min, n_neurons_irprop_min, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m)

    # iRPROP+
    rprop4_average_error_validation_arr, rprop4_average_accuracy_validation_arr, rprop4_average_error_testset, \
    rprop4_standard_deviation_error_testset, rprop4_average_accuracy_testset, rprop4_standard_deviation_accuracy_testset, \
    rprop4_average_convergence, rprop4_standard_deviation_convergence, rprop4_average_error_training_arr, rprop4_average_accuracy_training_arr \
        = test_rprop("irprop+", epoch, n_test, eta_p_irprop_plus, eta_m_irprop_plus, n_neurons_irprop_plus, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m)

    # Print learning result table for RPROP-
    print_result("RPROP-", rprop1_average_error_testset, rprop1_standard_deviation_error_testset,
                          rprop1_average_accuracy_testset, rprop1_standard_deviation_accuracy_testset,
                          rprop1_average_convergence, rprop1_standard_deviation_convergence, "Test-Set")

    # Print learning result table for RPROP+
    print_result("RPROP+", rprop2_average_error_testset, rprop2_standard_deviation_error_testset,
                          rprop2_average_accuracy_testset, rprop2_standard_deviation_accuracy_testset,
                          rprop2_average_convergence, rprop2_standard_deviation_convergence, "Test-Set")

    # Print learning result table for iRPROP-
    print_result("iRPROP-", rprop3_average_error_testset, rprop3_standard_deviation_error_testset,
                          rprop3_average_accuracy_testset, rprop3_standard_deviation_accuracy_testset,
                          rprop3_average_convergence, rprop3_standard_deviation_convergence, "Test-Set")

    # Print learning result table for iRPROP+
    print_result("iRPROP+", rprop4_average_error_testset, rprop4_standard_deviation_error_testset,
                          rprop4_average_accuracy_testset, rprop4_standard_deviation_accuracy_testset,
                          rprop4_average_convergence, rprop4_standard_deviation_convergence, "Test-Set")

    # Print comparison graph for error on validation set
    print_graph(rprop1_average_error_validation_arr, rprop2_average_error_validation_arr,
                rprop3_average_error_validation_arr, rprop4_average_error_validation_arr,
                names=["RPROP-", "RPROP+", "iRPROP-", "iRPROP+"], title='Validation Error', xlabel='Epoch',
                ylabel='Error')

    # Print comparison graph for accuracy on validation set
    print_graph(rprop1_average_accuracy_validation_arr, rprop2_average_accuracy_validation_arr,
                rprop3_average_accuracy_validation_arr, rprop4_average_accuracy_validation_arr,
                names=["RPROP-", "RPROP+", "iRPROP-", "iRPROP+"], title='Validation Accuracy', xlabel='Epoch',
                ylabel='Accuracy')

    # Print comparison graph for error on training set
    print_graph(rprop1_average_error_training_arr, rprop2_average_error_training_arr,
                rprop3_average_error_training_arr, rprop4_average_error_training_arr,
                names=["RPROP-", "RPROP+", "iRPROP-", "iRPROP+"], title='Training Error', xlabel='Epoch',
                ylabel='Error')

    # Print comparison graph for accuracy on training set
    print_graph(rprop1_average_accuracy_training_arr, rprop2_average_accuracy_training_arr,
                rprop3_average_accuracy_training_arr, rprop4_average_accuracy_training_arr,
                names=["RPROP-", "RPROP+", "iRPROP-", "iRPROP+"], title='Training Accuracy', xlabel='Epoch',
                ylabel='Accuracy')


"""
    Function to test a single version of rprop, with result on Test Set, Training Set and Validation Set
    
    arg:
        rprop_method -> variant of rprop
        epoch -> numbero of epoch for learning phase
        n_test -> number of test for each rprop variant
        eta_p -> value for eta_plus
        eta_m -> value for eta_minus
        n_neurons -> number of neurons for hidden layer
"""
def single_version_valutation(rprop_method, epoch, n_test, eta_p, eta_m, n_neurons):
    train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m = prepare_MNIST_data()

    rprop_average_error_validation_arr, rprop_average_accuracy_validation_arr, rprop_average_error_testset, \
    rprop_standard_deviation_error_testset, rprop_average_accuracy_testset, rprop_standard_deviation_accuracy_testset, \
    rprop_average_convergence, rprop_standard_deviation_convergence, rprop_average_error_training_arr, rprop_average_accuracy_training_arr \
        = test_rprop(rprop_method, epoch, n_test, eta_p, eta_m, n_neurons, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m)

    # Print result on Test-Set
    print_result(str(rprop_method), rprop_average_error_testset, rprop_standard_deviation_error_testset,
                          rprop_average_accuracy_testset, rprop_standard_deviation_accuracy_testset,
                          rprop_average_convergence, rprop_standard_deviation_convergence, "Test-Set")

    # Print comparison graph between error on validation set and training set
    print_graph(rprop_average_error_training_arr, rprop_average_error_validation_arr,
                names=["Training-set", "Validation-set"], title='Error evaluation' + str(rprop_method), xlabel='Epoch',
                ylabel='Error')

    # Print comparison graph between accuracy on validation set and training set
    print_graph(rprop_average_accuracy_training_arr, rprop_average_accuracy_validation_arr,
                names=["Training-set", "Validation-set"], title='Accuracy evaluation' + str(rprop_method), xlabel='Epoch',
                ylabel='Accuracy')


"""
    Function used to perform model selection, based on the gris-search technique
    
    arg:
        rprop_method -> selected rprop variant
        epoch -> numbero of epoch for learning phase
        n_test -> number of test for each rprop variant
    return:
        result_matrix_sorted -> result matrix ordered by accuracy on test set
            each row is a specific model, with relative measurement 
            columns: 
                - Number of hidden neurons 
                - eta+
                - eta-
                - average error on test-set
                - std deviation for error
                - average accuracy on test-set
                - std_deviation for accuracy
                - average epoch for convergence
                - std deviation for convergence
"""
def model_selection(rprop_method, epoch, n_test):
    train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m = prepare_MNIST_data()

    eta_plus_values = [1.2, 1.15, 1.1, 1.05]
    eta_minus_values = [0.5, 0.4, 0.3, 0.2]
    hidden_neurons_values = [150, 200, 250, 300]

    result_matrix = []

    for hidden_neurons in hidden_neurons_values:
        for eta_plus in eta_plus_values:
            for eta_minus in eta_minus_values:
                print(" Test with eta_plus = " + str(eta_plus) + "\teta_minus = " + str(eta_minus) + "\tHidden neurons = " + str(hidden_neurons))
                average_error_validation_arr, average_accuracy_validation_arr, average_error_testset, \
                standard_deviation_error_testset, average_accuracy_testset, standard_deviation_accuracy_testset, \
                average_convergence, standard_deviation_convergence, average_error_training_arr, average_accuracy_training_arr \
                    = test_rprop(rprop_method, epoch, n_test, eta_plus, eta_minus, hidden_neurons, train_m, train_label_m, valid_m, valid_label_m, test_m, test_label_m)
                print("\n")

                result_matrix.append(np.array([hidden_neurons, eta_plus, eta_minus, average_error_testset, standard_deviation_error_testset, average_accuracy_testset, standard_deviation_accuracy_testset, average_convergence, standard_deviation_convergence]))

    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    result_matrix = np.array(result_matrix)
    sorted_indices = np.argsort(result_matrix[:, 5])[::-1]
    result_matrix_sorted = result_matrix[sorted_indices]
    print("\n N_neurons, eta+, eta-, error_avg, std_dev_error, accuracy_avg, std_dev_accuracy, min_epoch_avg, std_dev_min_epoch\n")
    print(result_matrix_sorted)

    return result_matrix_sorted


if __name__ == '__main__':
    # Evaluation performance of different rprop variant, with test-set result - unique model, standard parameter
    """
    epoch = 2
    n_test = 1
    eta_plus = 1.2
    eta_minus = 0.5
    num_neurons = 150
    evaluation_rprop_version(epoch, n_test, eta_plus, eta_minus, num_neurons)
    """

    # Evaluation of irprop+ with training and validation error/accuracy graph - standard parameter
    """
    epoch = 100
    n_test = 1
    eta_plus = 1.2
    eta_minus = 0.5
    num_neurons = 150
    single_version_valutation("irprop+", epoch, n_test,  eta_plus, eta_minus, num_neurons)
    """

    # Selection of best model for rprop+
    """
    epoch = 100
    n_test = 5
    rprop_method = "rprop+"
    model_selection(rprop_method, epoch, n_test)
    """

    # Selection of best model irprop+
    """
    epoch = 100
    n_test = 5
    rprop_method = "irprop+"
    model_selection(rprop_method, epoch, n_test)
    """

    # Selection of best model for irprop-
    """
    epoch = 100
    n_test = 5
    rprop_method = "irprop-"
    model_selection(rprop_method, epoch, n_test)
    """

    # Selection of best model for rprop-
    """
    epoch = 100
    n_test = 5
    rprop_method = "rprop-"
    model_selection(rprop_method, epoch, n_test)
    """

    # Evaluation performance of different rprop variant, with test-set result
    epoch = 200
    n_test = 10

    # best model for RPROP-
    eta_p_rprop_min = 1.05
    eta_m_rprop_min = 0.4
    n_hidden_rprop_min = 300

    # best model for RPROP+
    eta_p_rprop_plus = 1.05
    eta_m_rprop_plus = 0.3
    n_hidden_rprop_plus = 300

    # best model for iRPROP-
    eta_p_irprop_min = 1.05
    eta_m_irprop_min = 0.5
    n_hidden_irprop_min = 300

    # best model for iRPROP+
    eta_p_irprop_plus = 1.05
    eta_m_irprop_plus = 0.5
    n_hidden_irprop_plus = 250

    eta_plus_list = [eta_p_rprop_min, eta_p_rprop_plus, eta_p_irprop_min, eta_p_irprop_plus]
    eta_minus_list = [eta_m_rprop_min, eta_m_rprop_plus, eta_m_irprop_min, eta_m_irprop_plus]
    num_neurons_list = [n_hidden_rprop_min, n_hidden_rprop_plus, n_hidden_irprop_min, n_hidden_irprop_plus]
    evaluation_rprop_version(epoch, n_test, eta_plus_list, eta_minus_list, num_neurons_list)


"""
 End File main.py
"""