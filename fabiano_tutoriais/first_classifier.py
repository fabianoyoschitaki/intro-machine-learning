#random
#import random

from scipy.spatial import distance

def euc(a, b):
	return distance.euclidean(a,b)

class FabianoScrappyKNN():
	def fit(self, X_train, y_train):
		#memorizing
		self.X_train = X_train
		self.y_train = y_train
	
	# Euclidean distance o measure the distance between 2 points
	def predict(self, X_test):
		predictions = []
		for row in X_test:
			#random
			#label = random.choice(self.y_train)
			label = self.closest(row)
			predictions.append(label)
		return predictions
	
	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist < best_dist:
				best_dist = dist
				best_index = i
		return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

#from sklearn.neighbors import KNeighborsClassifier
my_classifier = FabianoScrappyKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)