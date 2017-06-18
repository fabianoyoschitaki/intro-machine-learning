#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.
	
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
	
	You found in the SVM mini-project that the parameter tune can significantly speed up the training time of a machine learning algorithm. A general rule is that the parameters can tune the complexity of the algorithm, with more complex algorithms generally running more slowly.

	Another way to control the complexity of an algorithm is via the number of features that you use in training/testing. The more features the algorithm has available, the more potential there is for a complex fit. We will explore this in detail in the "Feature Selection" lesson, but you'll get a sneak preview now.

	What's the number of features in your data? (Hint: the data is organized into a numpy array where the number of rows is the number of data points and the number of columns is the number of features; so to extract this number, use a line of code like len(features_train[0]).)
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# There are two methods by which you can calculate the number of featuress

#1 - len(features_train) # default axis is 0 by which you are calculating the number of columns.
#2 - len(features_train.column) # it will calculate the number of columns(features)
print "Number of columns in features_train is ", len(features_train[0])
print "Default axis is 0 by which you are calculating the number of rows of features_train is ", len(features_train)

print "Number of columns in features_test is ", len(features_test[0])
print "Default axis is 0 by which you are calculating the number of rows of features_test is ", len(features_test)
#########################################################
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#Number of columns in features_train is  3785
#Default axis is 0 by which you are calculating the number of rows of features_train is  15820
#Number of columns in features_test is  3785
#Default axis is 0 by which you are calculating the number of rows of features_test is  1758
#
# With /tools/email_preprocess.py percentile = 1
#
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#Number of columns in features_train is  379
#Default axis is 0 by which you are calculating the number of rows of features_train is  15820
#Number of columns in features_test is  379
#Default axis is 0 by which you are calculating the number of rows of features_test is  1758
#########################################################