"""
==================================================================
ADASYN Over-sampling Method applied to real financial transactions
==================================================================
An illustration of the Adaptive Synthetic Sampling Approach for Imbalanced
Learning ADASYN method applied to real financial transactions for STIRP project
"""

# Authors: Fabiano Yoschitaki <fabianoyoschitaki@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import ADASYN
from time import time
from datetime import datetime

print(__doc__)

date_time_str = datetime.now().strftime("%Y%m%d%H%M%S")
input_csv_file_name = 'TB_CORRELATION_ANALYSIS_CREDSYSTEM.csv'
txt_file_name = "adasyn_" + date_time_str + "_info.txt"
output_csv_file_name = "adasyn_" + date_time_str + "_resampled.csv"
output_text_file = open(txt_file_name,"w") 
output_text_file.write(__doc__)

output_text_file.write("Starting...\n\n")

# Load transactions from csv file. Skipping first row (header data)
print "Loading csv [", input_csv_file_name, "] task started..."
output_text_file.write("Loading csv [" + input_csv_file_name + "] task started...\n")

t_load_start = time()
data = np.loadtxt(input_csv_file_name, delimiter=';',skiprows=1, dtype="int")
t_load_end = round(time()-t_load_start, 3)
print "Loading csv [", input_csv_file_name, "] task took:", t_load_end, "s"
output_text_file.write("Loading csv [" + input_csv_file_name + "] task took:" + str(t_load_end) + "s\n\n")
print

# We want to extract column 14 FL_FRAUDE to y array
idx_y_colum = [14] 
# We want all columns except 14 to be in X array
idx_X_columns = [i for i in xrange(np.shape(data)[1]) if i not in idx_y_colum] 
# Transaction values
X = data[:,idx_X_columns].astype(int) 
# FLG_FRAUD contains only 0 (non-fraud transaction) or 1 (fraud transaction) values. ravel() transforms 2dimensions to 1dimension
y = data[:,idx_y_colum].astype(int).ravel() 

print "Transaction data + Flag Fraud (15 columns)"
output_text_file.write("Transaction data + Flag Fraud (15 columns)\n")
print "type(data):", type(data)
output_text_file.write("type(data):" + str(type(data)) + "\n")
print "type(data[0][0]):", type(data[0][0])
output_text_file.write("type(data[0][0]):" + str(type(data[0][0])) + "\n")
print "data.shape:", data.shape
output_text_file.write("data.shape:" + str(data.shape) + "\n\n")
print
print "Only Transaction data (14 columns)"
output_text_file.write("Only Transaction data (14 columns)\n")
print "type(X):", type(X)
output_text_file.write("type(X):" + str(type(X)) + "\n")
print "type(X[0][0]):", type(X[0][0])
output_text_file.write("type(X[0][0]):" + str(type(X[0][0])) + "\n")
print "X.shape:", X.shape
output_text_file.write("X.shape:" + str(X.shape) + "\n\n")
print
print "Only Flag Fraud data (1 column, the last one)"
output_text_file.write("Only Flag Fraud data (1 column, the last one)\n")
print "type(y):", type(y)
output_text_file.write("type(y):" + str(type(y)) + "\n")
print "type(y[0]):", type(y[0])
output_text_file.write("type(y[0]):" + str(type(y[0])) + "\n")
print "y.shape:", y.shape
output_text_file.write("y.shape:" + str(y.shape) + "\n\n")
print

# Instanciate a PCA object for the sake of easy visualisation
pca = PCA(n_components=2)

# Fit and transform x to visualise inside a 2D feature space
X_vis = pca.fit_transform(X)

# Apply the random over-sampling method ADASYN
ada = ADASYN()

print "ADASYN resampling task started..."
output_text_file.write("ADASYN resampling task started...\n")
t_resampling_start = time()
X_resampled, y_resampled = ada.fit_sample(X, y)
t_resampling_end = round(time()-t_resampling_start, 3)
print "ADASYN resampling task took:", t_resampling_end, "s"
output_text_file.write("ADASYN resampling task took:" + str(t_resampling_end) + "s\n\n")
print

# X_resampled returns as float type. Transforming to int type
X_resampled = X_resampled.astype(int)

X_res_vis = pca.transform(X_resampled)

print "Only Transaction data (14 columns) oversampled"
output_text_file.write("Only Transaction data (14 columns) oversampled\n")
print "type(X_resampled):", type(X_resampled)
output_text_file.write("type(X_resampled):" + str(type(X_resampled)) + "\n")
print "type(X_resampled[0][0]):", type(X_resampled[0][0])
output_text_file.write("type(X_resampled[0][0]):" + str(type(X_resampled[0][0])) + "\n")
print "X_resampled.shape:", X_resampled.shape
output_text_file.write("X_resampled.shape:" + str(X_resampled.shape) + "\n\n")
print
print "Only Flag Fraud data (1 column) oversampled"
output_text_file.write("Only Flag Fraud data (1 column) oversampled\n")
print "type(y_resampled):", type(y_resampled)
output_text_file.write("type(y_resampled):" + str(type(y_resampled)) + "\n")
print "type(y_resampled[0]):", type(y_resampled[0])
output_text_file.write("type(y_resampled[0]):" + str(type(y_resampled[0])) + "\n")
print "y_resampled.shape:", y_resampled.shape
output_text_file.write("y_resampled.shape:" + str(y_resampled.shape) + "\n\n")
print

# transforms y_resampled as vector to [][] and append to 14 column 2d numpy transaction data
resampled_data = np.append(X_resampled, y_resampled[:, None], 1) 

print "Transaction data + Flag Fraud (15 columns) oversampled"
output_text_file.write("Transaction data + Flag Fraud (15 columns) oversampled\n")
print "type(resampled_data):", type(resampled_data)
output_text_file.write("type(resampled_data):" + str(type(resampled_data)) + "\n")
print "type(resampled_data[0][0]):", type(resampled_data[0][0])
output_text_file.write("type(resampled_data[0][0]):" + str(type(resampled_data[0][0])) + "\n")
print "resampled_data.shape:", resampled_data.shape
output_text_file.write("resampled_data.shape:" + str(resampled_data.shape) + "\n\n")
print

# Save to file overriding float output type to string, delimiting with ; character like the original file
print "Saving output resampled data task..."
output_text_file.write("Saving output resampled data task...\n")
t_save_start = time()
np.savetxt(output_csv_file_name, resampled_data, delimiter=";", fmt="%s")
t_save_end = round(time()-t_save_start, 3)
print "Saving output resampled data task took:", t_save_end, "s"
output_text_file.write("Saving output resampled data task took:" + str(t_save_end) + "s\n\nSuccess")
print

output_text_file.close()

# Print all values for debug
#print "------Original------"
#for i in range(0, X.shape[0]):
#    print "(",i,") ", str(X[i][0]),",",str(X[i][1]),",",str(X[i][2]),",",str(X[i][3]),",",str(X[i][4]),",",str(X[i][5]),",",str(X[i][6]),",",str(X[i][7]),",",str(X[i][8]),",",str(X[i][9]),",",str(X[i][10]),",",str(X[i][11]),",",str(X[i][12]),",",str(X[i][13]),"->",str(y[i])
#print "------Original------"
#print "------Resampled------"
#for i in range(0, X_resampled.shape[0]):
#    print "(",i,") ", str(X_resampled[i][0]),",",str(X_resampled[i][1]),",",str(X_resampled[i][2]),",",str(X_resampled[i][3]),",",str(X_resampled[i][4]),",",str(X_resampled[i][5]),",",str(X_resampled[i][6]),",",str(X_resampled[i][7]),",",str(X_resampled[i][8]),",",str(X_resampled[i][9]),",",str(X_resampled[i][10]),",",str(X_resampled[i][11]),",",str(X_resampled[i][12]),",",str(X_resampled[i][13]),"->",str(y_resampled[i])
#print "------Resampled------"

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)

c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Fraude", alpha=0.5)
c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Nao-fraude", alpha=0.5)
ax1.set_title('Original')
ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1], label="Fraude", alpha=0.5)
ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1], label="Nao-fraude", alpha=0.5)

#second axe
ax2.set_title('ADASYN oversampling')

# plotting
for ax in (ax1, ax2):
	print ax
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.spines['left'].set_position(('outward', 10))
	ax.spines['bottom'].set_position(('outward', 10))
	ax.set_xlim([-6, 8])
	ax.set_ylim([-6, 6])

plt.figlegend((c0, c1), ('Fraude', 'Nao-fraude'), loc='lower center', ncol=2, labelspacing=0.0)
plt.tight_layout(pad=3)
plt.show()
