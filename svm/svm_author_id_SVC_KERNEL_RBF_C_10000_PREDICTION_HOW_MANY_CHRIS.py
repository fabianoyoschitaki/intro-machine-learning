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
print "SVC kernel rbf com C = 10000, resultados de teste para Chris (classe 1)"

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

from collections import Counter
C = Counter(pred)

for key, value in C.iteritems():
	print "classe ", key, ' tem quantidade: ', value

#########################################################
#output

#SVC kernel rbf com C = 10000, resultados de teste para Chris (classe 1)
#tempo de treinamento C=10000.0: 115.312 s
#tempo de testes C=10000.0: 11.645 s
#tempo de score C=10000.0: 11.7 s
#Accuracy C=10000.0: 0.990898748578
#classe  0  tem quantidade:  881
#classe  1  tem quantidade:  877