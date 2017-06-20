#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

# import the classifiers
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(features_train, labels_train)

# print the accuracy and display the decision boundary
accuracy = '{0}'.format(clf.score(features_test, labels_test))
prettyPicture(clf, features_test, labels_test)

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.title("kNN Classifier " + accuracy)
plt.scatter(bumpy_fast, grade_fast, color = "y", label="rapido")
plt.scatter(grade_slow, bumpy_slow, color = "g", label="devagar")
plt.legend()
plt.xlabel("Bumpiness")
plt.ylabel("Grade")
plt.show()

################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

print "FIM"