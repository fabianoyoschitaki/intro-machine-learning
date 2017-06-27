import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

print "\ntype(iris)"
print type(iris)

print "\niris.feature_names"
print iris.feature_names

print "\niris.target_names"
print iris.target_names

print "\niris.data[0]"
print iris.data[0]

print "\niris.target[0]"
print iris.target[0]

print "\niris.target_names[iris.target[0]]"
print iris.target_names[iris.target[0]]
print

# iterating over the 150 rows

#for i in range(len(iris.target)):
#	print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

# remove one example of each type of flower
test_idx = [0, 50, 100] # setosa, versicolor and so on

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

print "\ntype(train_target)"
print type(train_target)
print "\ntrain_target.shape"
print train_target.shape
print "\ntype(train_data)"
print type(train_data)
print "\ntrain_data.shape"
print train_data.shape

print "\ntype(test_target)"
print type(test_target)
print "\ntest_target.shape"
print test_target.shape
print "\ntype(test_data)"
print type(test_data)
print "\ntest_data.shape"
print test_data.shape

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print "\nActual Answer for test_data: ", test_target
print "Decision Tree predicts: ", clf.predict(test_data)
print "Decision Tree accuracy: ", clf.score(test_data, test_target)

# Visualize the tree
# viz code
#from sklearn.externals.six import StringIO
#import pydotplus
#dot_data = StringIO()
#tree.export_graphviz(clf, 
#	out_file=dot_data, 
#	feature_names=iris.feature_names,
#	class_names=iris.target_names,
#	filled=True,
#	rounded=True,
#	impurity=False)
#
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("iris.pdf")
	


