#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

from sklearn.svm import SVC

clf = SVC(kernel="linear")

t0 = time()
clf.fit(features_train, labels_train)  
print "tempo de treinamento:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de testes:", round(time()-t1, 3), "s"

t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score:", round(time()-t2, 3), "s"

print "Accuracy:", accuracy

#########################################################

#output

#tempo de treinamento: 225.684 s
#tempo de testes: 24.04 s
#tempo de score: 23.552 s
#Accuracy: 0.984072810011

