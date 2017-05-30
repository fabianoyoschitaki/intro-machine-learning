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
print "SVC kernel rbf com 1% do features e labels de treino. C = 10, 100, 1000 e 10000"

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf = SVC(kernel="rbf", C=1.0)
t0 = time()
clf.fit(features_train, labels_train)  
print "tempo de treinamento C=1.0:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "tempo de testes C=1.0:", round(time()-t1, 3), "s"
t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score C=1.0:", round(time()-t2, 3), "s"
print "Accuracy C=1.0:", accuracy
print ""

clf = SVC(kernel="rbf", C=10.0)
t0 = time()
clf.fit(features_train, labels_train)  
print "tempo de treinamento C=10.0:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "tempo de testes C=10.0:", round(time()-t1, 3), "s"
t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score C=10.0:", round(time()-t2, 3), "s"
print "Accuracy C=10.0:", accuracy
print ""

clf = SVC(kernel="rbf", C=100.0)
t0 = time()
clf.fit(features_train, labels_train)  
print "tempo de treinamento C=100.0:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "tempo de testes C=100.0:", round(time()-t1, 3), "s"
t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score C=100.0:", round(time()-t2, 3), "s"
print "Accuracy C=100.0:", accuracy
print ""

clf = SVC(kernel="rbf", C=1000.0)
t0 = time()
clf.fit(features_train, labels_train)  
print "tempo de treinamento C=1000.0:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "tempo de testes C=1000.0:", round(time()-t1, 3), "s"
t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score C=1000.0:", round(time()-t2, 3), "s"
print "Accuracy C=1000.0:", accuracy
print ""

clf = SVC(kernel="rbf", C=10000.0)
t0 = time()
clf.fit(features_train, labels_train)  
print "tempo de treinamento C=10000.0:", round(time()-t0, 3), "s"
t1 = time()
pred = clf.predict(features_test)
print "tempo de testes C=10000.0:", round(time()-t1, 3), "s"
t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score C=10000.0:", round(time()-t2, 3), "s"
print "Accuracy C=10000.0:", accuracy
print ""

#########################################################
#output

#SVC kernel rbf com 1% do features e labels de treino. C = 10, 100, 1000 e 10000
#tempo de treinamento C=1.0: 0.109 s
#tempo de testes C=1.0: 1.199 s
#tempo de score C=1.0: 1.157 s
#Accuracy C=1.0: 0.616040955631
#
#tempo de treinamento C=10.0: 0.11 s
#tempo de testes C=10.0: 1.174 s
#tempo de score C=10.0: 1.172 s
#Accuracy C=10.0: 0.616040955631
#
#tempo de treinamento C=100.0: 0.125 s
#tempo de testes C=100.0: 1.154 s
#tempo de score C=100.0: 1.171 s
#Accuracy C=100.0: 0.616040955631
#
#tempo de treinamento C=1000.0: 0.11 s
#tempo de testes C=1000.0: 1.127 s
#tempo de score C=1000.0: 1.115 s
#Accuracy C=1000.0: 0.821387940842
#
#tempo de treinamento C=10000.0: 0.115 s
#tempo de testes C=10000.0: 0.924 s
#tempo de score C=10000.0: 0.938 s
#Accuracy C=10000.0: 0.892491467577