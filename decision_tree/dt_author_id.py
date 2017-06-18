#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.
	
    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
	
	http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import tree 
clf = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
clf.fit(features_train, labels_train)
print "tempo de treinamento:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "tempo de testes:", round(time()-t1, 3), "s"

t2 = time()
accuracy = clf.score(features_test, labels_test)
print "tempo de score:", round(time()-t2, 3), "s"

print "Accuracy of DT with min_samples_split = 40 is ", accuracy
#########################################################
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#tempo de treinamento: 51.268 s
#tempo de testes: 0.047 s
#tempo de score: 0.031 s
#Accuracy of DT with min_samples_split = 40 is  0.978384527873
#
# With /tools/email_preprocess.py percentile = 1
#
#no. of Chris training emails: 7936
#no. of Sara training emails: 7884
#tempo de treinamento: 4.359 s
#tempo de testes: 0.0 s
#tempo de score: 0.016 s
#Accuracy of DT with min_samples_split = 40 is  0.967007963595
#########################################################

