"""
 In this file there are defined functions for download and manipulate data from MNIST dataset
 MNIST is a dataset of 70000 handwritten digits, each one represented by a 28x28 matrix of 0-255 value

 File: dataset.py
 Author: Pastore Luca N97000431
 Target: Università degli studi di Napoli Federico II
"""

from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import scipy.ndimage


"""
    Function tha allow to download raw images contains in MNIST dataset
    Every image is represented  by a 28x28 matrix. Each matrix value is in the range [0, 255]
    
    return:
        train_X -> 60000 images for training set
        train_y -> 60000 labels for training set image
        test_X -> 10000 images for test set
        test_y -> 10000 labels for test set image
"""
def download_data():
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    #print('X_train: ' + str(train_X.shape))
    #print('Y_train: ' + str(train_y.shape))
    #print('X_test:  ' + str(test_X.shape))
    #print('Y_test:  ' + str(test_y.shape))

    return (train_X, train_y), (test_X, test_y)


"""
    Function that allow to print a raw image from MNIST dataset
    
    arg:
        image -> image of digit to print
"""
def print_image(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


"""
    Function that print MNIST image. One image for each digit       
"""
def print_digits(set_X):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        print_image(set_X[i])


"""
    Function that split element of MNIST dataset in Training Set and Validation Set
    
    arg: 
        set_X -> training set of MNIST image
    return:
        training_set -> elements for training
        training_set_labels -> labels for training elements
        validation_set -> elements for validation
        validation_labels -> labels for validation elements
"""
def split_training_set(train_X, train_y):
    tot_elem = len(train_X)
    num_elem_training_set = 12500

    training_set = train_X[:num_elem_training_set]
    training_set_labels = train_y[:num_elem_training_set]

    #validation_set = train_X[num_elem_training_set: tot_elem]
    #validation_labels = train_y[num_elem_training_set: tot_elem]

    validation_set = train_X[num_elem_training_set: num_elem_training_set+2500]
    validation_labels = train_y[num_elem_training_set: num_elem_training_set+2500]

    return (training_set, training_set_labels), (validation_set, validation_labels)


"""
    Function used to commute a matrix in an array
    
    arg:
        x -> matrix NxM
    return 
        flat_X -> array of N*M elements
"""
def flatter_element(X):
    flat_X = []

    for i in range(len(X)):
        flat_X.append(X[i].flatten())

    return np.array(flat_X)


"""
    Function used to normalize every pixel value for a set of raw image 
    Each pixel is reported in a range [0, 1] instead of [0, 255]
    
    arg: 
        x -> matrix of images, every row is an image representing by an array contains value for each pixel
    return:
        matrix normalized to [0, 1]
"""
def normalize_pixel_value(x):
    f_min = np.min(x)
    f_max = np.max(x)

    return (x - f_min) / (f_max - f_min)


"""
    Function used to encode the target array using one-hot encoding
    
    arg:
        y -> target array
    return 
        one_hot_y -> matrix Nx10, where N is the number of elements of dataset and 10 is the numbers of classes
"""
def one_hot_encode(y):
    one_hot_y = []

    for i in range(len(y)):
        tmp = np.zeros(10)
        tmp[y[i]] = 1
        one_hot_y.append(tmp)

    return np.array(one_hot_y)


"""
    Function that allow to scale raw im dimension from 28x28 to 14x14

    arg:
        X -> dataset of MNIST image, array of matrix 28x28 
    return:
        scaled_X -> dataset of MNIST image, array of matrix 14x14
"""
def scale_images_dimension(X):
    zoom_factors = (0.5, 0.5)
    scaled_X = []

    for i in range(len(X)):
        scaled_image = scipy.ndimage.zoom(X[i], zoom_factors)
        scaled_X.append(scaled_image)

    return np.array(scaled_X)


"""
    Function used to prepare MNIST data for network processing.
    For each image:
    - corresponding matrix is commuted to an array
    - pixel value are normalized to range [0, 1]
    - label array is encoding with One-Hot-Encoding
    
    arg:
        X -> dataset of MNIST image, array of matrix 28x28
        y -> array of target for each image 
    return:
        normalized_x -> matrix Nx784, where N is the numbers of elements and each row represent an image
        vector_y -> matrix Nx10, where N is the numbers of elements and 10 are the number of class (10 digits) 
"""
def manipule_data(X, y):
    scaled_X = scale_images_dimension(X)
    flat_X = flatter_element(scaled_X)
    normalized_x = normalize_pixel_value(flat_X)
    vector_y = one_hot_encode(y)

    return normalized_x, vector_y


"""
 End File dataset.py
"""