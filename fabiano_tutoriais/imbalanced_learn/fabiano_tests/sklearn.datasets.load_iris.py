"""
=============================================
Let's say you are interested in the samples 10, 25, and 50, and want to know their class name.
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py
=============================================
"""
print(__doc__)
from sklearn.datasets import load_iris
dataset = load_iris()

print("\ndataset.DESCR")
print(dataset.DESCR)

""" pega o elemento 11, 26 e 51 """
print("\ndataset.data[[10, 25, 50]]")
print(dataset.data[[10, 25, 50]])

print("\ndataset.target[[10, 25, 50]]")
print(dataset.target[[10, 25, 50]])

print("\nlist(dataset.target[[10, 25, 50]])")
print(list(dataset.target[[10, 25, 50]]))

print("\nlist(dataset.target_names)")
print(list(dataset.target_names))

print("\nlist(dataset.feature_names)")
print(list(dataset.feature_names))

print("\ndataset.target_names")
print(dataset.target_names)
