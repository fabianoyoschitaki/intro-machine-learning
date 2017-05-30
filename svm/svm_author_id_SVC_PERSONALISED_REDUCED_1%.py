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
print "SVC personalizado 1% do features e labels de treino"
clf = SVC()

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
	decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
	max_iter=-1, probability=False, random_state=None, shrinking=True,
	tol=0.001, verbose=False)

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

#SVC personalizado 1% do features e labels de treino
#tempo de treinamento: 0.172 s
#tempo de testes: 1.536 s
#tempo de score: 1.555 s
#Accuracy: 0.616040955631

