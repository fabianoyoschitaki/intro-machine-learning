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
print "SVC kernel rbf com 1% do features e labels de treino. C = 10000, resultados do registro 10, 26 e 50"

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

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
print "Classe do registro 10:", pred[10]
print "Classe do registro 26:", pred[26]
print "Classe do registro 50:", pred[50]


#########################################################
#output

#SVC kernel rbf com 1% do features e labels de treino. C = 10000, resultados do registro 10, 26 e 50
#tempo de treinamento C=10000.0: 0.108 s
#tempo de testes C=10000.0: 0.926 s
#tempo de score C=10000.0: 0.96 s
#Accuracy C=10000.0: 0.892491467577
#Classe do registro 10: 1
#Classe do registro 26: 0
#Classe do registro 50: 1