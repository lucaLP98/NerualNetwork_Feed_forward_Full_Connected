# NerualNetwork_Feed_forward_Full_Connected
In this project, carried out as part of the master's degree in Computer Science at the University of Naples Federico II, we developed a Feed-Forward Full-Connected Multi-Layer Neural Network from scratch, using the Python language (in version 3.10) and only the NumPy library.
In the developed library, an FNN nerual network can be instantiated, establishing the number of layers, the number of neurons for each layer, the activation function for each layer, and the error function to be used during the learning phase.

<h2>Project specifications</h2>
Design and implementation of functions to simulate the forward propagation of a multi-layer neural network. 
Possibility to implement networks with more than one layer of internal nodes and with any activation function for each layer.
<br>
Design and implementation of functions for the realization of back-propagation for multi-layer neural networks, for any choice of activation function for network nodes and the possibility of using sum-of-squares or cross-entropy (with and without soft-max) as the error function.
<br><br>
Implementation of the four variants of RPROP reported in the article "<b>Empirical evaluation of the improved Rprop learning algorithms, Christian Igel, Michael Husken, neurocomputing, 2003</b>" as an algorithm for updating network parameters during the learning phase:
  <ul>
    <li>RPROP without weight-backtracking (RPROP-)</li>
    <li>RPROP with weight-backtracking (RPROP+)</li>
    <li>Improved RPROP with weight-backtracking (iRPROP+)</li>
    <li>Improved RPROP without weight-backtracking (iRPROP-)</li>
  </ul>

<h2>Purpose of the project</h2>
The project aims to study and understand how an FNN neural network works, implementing the latter without the use of third-party libraries developed specifically for Machine-Learning, such as TesnorFlow and PyTorch.
<br><br>
The performance of the implemented FNN is then extensively analyzed, performing a multiclass classification task, using the MNIST dataset. Classical resilient backpropagation (RProp) is compared with the different variants reported in the article and implemented in this library. Evaluations are made using a HOLD-OUT approach, and the following are used as terms of comparison: average error on Test-Set, average accuracy on Test-Set, epochs required for convergence during the Learning phase. The GRID-SEARCH approach is, in addition, used to determine the best hyper-parameters for each RProp variant.

<h2>Project Documentation</h2>
Project documentation can be viewed at the following links:
<ul>
    <li>Italian Version: <a href="https://drive.google.com/file/d/1DpmoJqMSKNFT_wce9zEmYNVoCaksTkjH/view?usp=drive_link">doc link</a></li>
    <li>English version: <a href="">doc link</a></li>
</ul>
