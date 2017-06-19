"""
======
ADASYN Method
======

An illustration of the Adaptive Synthetic Sampling Approach for Imbalanced
Learning ADASYN method.

"""

# Authors: Christos Aridas
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
import numpy as np
from imblearn.over_sampling import ADASYN

print(__doc__)

# Generate the dataset
#X, y = make_classification(
#	n_classes=2, 
#	class_sep=2, 
#	weights=[0.1, 0.9], #fraude, nao fraude
#	n_informative=3, 
#	n_redundant=1, 
#	flip_y=0,
#	n_features=5, 
#	n_clusters_per_class=1,
#	n_samples=100, 
#	random_state=10)
#X = <type 'numpy.ndarray'>
#y = <type 'numpy.ndarray'>
#X.shape = valor de n_samples, valor de n_features (tamanho dim1 e dim2)
#y.shape = valor de n_samples
#print "type(X):", type(X)
#print "type(X[0][0]):", type(X[0][0])
#print "X.shape:", X.shape
#print
#print "type(y):", type(y)
#print "type(y[0]):", type(y[0])
#print "y.shape:", y.shape
#print 
#X.dtype = float64
#y.dtype = int32
#X.size = n_samples * valor de n_features
#y.size = n_samples
#X.ndim = 2
#y.ndim = 1
#for i in range(0, X.shape[0]):
#    print "(",i,") ", str(X[i][0]),",",str(X[i][1]),",",str(X[i][2]),",",str(X[i][3]),",",str(X[i][4]),"->",str(y[i])

# Load transactions
data = np.loadtxt('TB_CORRELATION_ANALYSIS_CREDSYSTEM.csv', delimiter=';',skiprows=1)
idx_OUT_columns = [14] # We want column 14 FL_FRAUDE to y array
idx_IN_columns = [i for i in xrange(np.shape(data)[1]) if i not in idx_OUT_columns] # We want all columns except 14 to be in X array
X = data[:,idx_IN_columns].astype(int) #transacoes contains all other columns
y = data[:,idx_OUT_columns].astype(bool).ravel() # array_fl_fraude contains only 0 (transaction ok) and 1 (transaction is fraud) values
print "type(X):", type(X)
print "type(X[0][0]):", type(X[0][0])
print "X.shape:", X.shape
print
print "type(y):", type(y)
print "type(y[0]):", type(y[0])
print "y.shape:", y.shape
print

#print X[1646][0], ",", X[1646][1], ",", X[1646][2], ",", X[1646][3], ",", X[1646][4], ",", X[1646][5], ",", X[1646][6], ",", X[1646][7], ",", X[1646][8], ",", X[1646][9], ",", X[1646][10], ",", X[1646][11], ",", X[1646][12], ",", X[1646][13], "->", y[1646]
#print X[1645][0], ",", X[1645][1], ",", X[1645][2], ",", X[1645][3], ",", X[1645][4], ",", X[1645][5], ",", X[1645][6], ",", X[1645][7], ",", X[1645][8], ",", X[1645][9], ",", X[1645][10], ",", X[1645][11], ",", X[1645][12], ",", X[1645][13], "->", y[1645]
#print
    
# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)

# Fit and transform x to visualise inside a 2D feature space
## X_vis = pca.fit_transform(X)

# Apply the random over-sampling
ada = ADASYN()

X_resampled, y_resampled = ada.fit_sample(X, y)
## X_res_vis = pca.transform(X_resampled)

print "X_resampled.shape:", X_resampled.shape
print "y_resampled.shape:", y_resampled.shape
print

#for i in range(0, X_resampled.shape[0]):
#    print "(",i,") ", str(X_resampled[i][0]),",",str(X_resampled[i][1]),",",str(X_resampled[i][2]),",",str(X_resampled[i][3]),",",str(X_resampled[i][4]),"->",str(y_resampled[i])

# Two subplots, unpack the axes array immediately
## f, (ax1, ax2) = plt.subplots(1, 2)

## c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5)
## c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5)
## ax1.set_title('Original')
## ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1], label="Class #0", alpha=0.5)
## ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1], label="Class #1", alpha=0.5)
#second axe
## ax2.set_title('ADASYN oversampling')

# make nice plotting
## for ax in (ax1, ax2):
##     ax.spines['top'].set_visible(False)
##     ax.spines['right'].set_visible(False)
##     ax.get_xaxis().tick_bottom()
##     ax.get_yaxis().tick_left()
##     ax.spines['left'].set_position(('outward', 10))
##     ax.spines['bottom'].set_position(('outward', 10))
##     ax.set_xlim([-6, 8])
##     ax.set_ylim([-6, 6])
## 
## plt.figlegend((c0, c1), ('Fraude', 'Nao-fraude'), loc='lower center', ncol=2, labelspacing=0.0)
## plt.tight_layout(pad=3)
## plt.show()
