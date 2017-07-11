# http://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html

# www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/?couponCode=KDNUGGETSPY

# The newest version (0.18) was just released a few days ago and now has built in support for Neural Network models. 
# In this article we will learn how Neural Networks work and how to implement them with the Python programming language and latest version of SciKit-Learn

# Neural Networks
 
# Neural Networks are a machine learning framework that attempts to mimic the learning pattern of natural biological neural networks. 
# Biological neural networks have interconnected neurons with dendrites that receive inputs, then based on these inputs they produce 
# an output signal through an axon to another neuron. We will try to mimic this process through the use of Artificial Neural Networks (ANN), 
# which we will just refer to as neural networks from now on. The process of creating a neural network begins with the most basic form, a single perceptron.

# The Perceptron
 
# Let's start our discussion by talking about the Perceptron! A perceptron has one or more inputs, a bias, an activation function, and a single output. 
# The perceptron receives inputs, multiplies them by some weight, and then passes them into an activation function to produce an output. 
# There are many possible activation functions to choose from, such as the logistic function, a trigonometric function, a step function etc. 
# We also make sure to add a bias to the perceptron, this avoids issues where all inputs could be equal to zero (meaning no multiplicative 
# weight would have an effect). Check out the diagram below for a visualization of a perceptron:

# [Perceptron]
#
# 			Bias 	---- w ---->   ___________
# Inputs 	x1 		---- w1 --->  /           \ 
#			x2 		---- w2 ---> | activation | ---------> output
#			x3 		---- w3 ---> |  function  |
#			x.. 	---- w..--->  \___________/
#

# Once we have the output we can compare it to a known label and adjust the weights accordingly (the weights usually start off with random initialization values). 
# We keep repeating this process until we have reached a maximum number of allowed iterations, or an acceptable error rate.

# To create a neural network, we simply begin to add layers of perceptrons together, creating a multi-layer perceptron model of a neural network. 
# You'll have an input layer which directly takes in your feature inputs and an output layer which will create the resulting outputs. Any layers in 
# between are known as hidden layers because they don't directly "see" the feature inputs or outputs. For a visualization of this check out the diagram below

# [Data]
 
# We'll use SciKit Learn's built in Breast Cancer Data Set which has several features of tumors with a labeled class indicating whether the tumor was 
# Malignant or Benign. We will try to create a neural network model that can take in these features and attempt to predict malignant or benign labels for 
# tumors it has not seen before. Let's go ahead and start by getting the data!

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data.target[[10, 50, 85]]
list(data.target_names)

print "type(data):", type(data) #<class 'sklearn.datasets.base.Bunch'>
print "data.keys():", data.keys()
print "data.DESCR:", data.DESCR
print "data.target_names:", data.target_names
print "data.feature_names:", data.feature_names
print "data.data:", data.data
print "data.target:", data.target


# This object is like a dictionary, it contains a description of the data and the features and targets:
# dict_keys(['DESCR', 'feature_names', 'target_names', 'target', 'data'])

# Let's set up our Data and our Labels:

# [Train Test Split] 
# Let's split our data into training and testing sets, this is done easily with SciKit Learn's train_test_split function from model_selection:

# [Data Preprocessing]

# The neural network may have difficulty converging before the maximum number of iterations allowed if the data is not normalized. 
# Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. Note that you must apply the same scaling 
# to the test set for meaningful results. There are a lot of different methods for normalization of data, we will use the built-in StandardScaler 
# for standardization.

# StandardScaler(copy=True, with_mean=True, with_std=True)

# [Training the model]

# Now it is time to train our model. SciKit Learn makes this incredibly easy, by using estimator objects. In this case we will import our estimator 
# (the Multi-Layer Perceptron Classifier model) from the neural_network library of SciKit-Learn! Next we create an instance of the model, 
# there are a lot of parameters you can choose to define and customize here, we will only define the hidden_layer_sizes. For this parameter you pass 
# in a tuple consisting of the number of neurons you want at each layer, where the nth entry in the tuple represents the number of neurons in the nth layer 
# of the MLP model. There are many ways to choose these numbers, but for simplicity we will choose 3 layers with the same number of neurons as there are 
# features in our data set:

# Now that the model has been made we can fit the training data to our model, remember that this data has already been processed and scaled:










