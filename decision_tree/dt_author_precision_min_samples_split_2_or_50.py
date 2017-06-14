"""
    Precisao da Arvore de Decisao com min_samples_split = 2 e 50
"""
    
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()
########################## DECISION TREE #################################
### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively

from sklearn import tree 
dtc_min_samples_split_2 = tree.DecisionTreeClassifier(min_samples_split=2)
dtc_min_samples_split_2.fit(features_train, labels_train)
acc_min_samples_split_2 = dtc_min_samples_split_2.score(features_test, labels_test)

dtc_min_samples_split_50 = tree.DecisionTreeClassifier(min_samples_split=50)
dtc_min_samples_split_50.fit(features_train, labels_train)
acc_min_samples_split_50 = dtc_min_samples_split_50.score(features_test, labels_test)

print "acc_min_samples_split_2:", acc_min_samples_split_2
print "acc_min_samples_split_50:", acc_min_samples_split_50

def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}
	
## Retorno:
##
## acc_min_samples_split_2: 0.908
## acc_min_samples_split_50: 0.912
## {"message": "{'acc_min_samples_split_50': 0.912, 'acc_min_samples_split_2': 0.908}"}

