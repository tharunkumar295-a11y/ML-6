# Experiment 6 — Principal Component Analysis (PCA) on Iris

# AIM

Reduce 4D Iris data to 2 principal components and visualize species separation.

# ALGORITHM

PCA via SVD: compute components that capture maximal variance. 

# DETAILED EXPLANATION

1. Load Iris dataset. 
2. PCA(n_components=2) computes two principal components. 
3. fit_transform centers data and projects onto top components. 
4. explained_variance_ratio_ shows how much variance each PC captures. 
5. Plot projected data and color by species.

# PROGRAM

from sklearn.decomposition import PCA 

from sklearn.datasets import load_iris 

import matplotlib.pyplot as plt 

import numpy as np 

iris = load_iris() 

X = iris.data; y = iris.target 

pca = PCA(n_components=2) 

Z = pca.fit_transform(X) 

print('Explained variance ratio:', 

np.round(pca.explained_variance_ratio_,4)) 

plt.scatter(Z[:,0], Z[:,1], c=y, cmap='viridis'); plt.xlabel('PC1'); 

plt.ylabel('PC2'); plt.title('PCA of Iris'); plt.show()

# SAMPLE OUTPUT

<img width="846" height="592" alt="image" src="https://github.com/user-attachments/assets/c6e88484-2947-4d9e-b970-eb770d561392" />

# RESULT

Students see PC1 captures most variance; PCA useful for visualization and dimensionality reduction. 

# THEORY

PCA finds orthogonal directions of maximal variance (eigenvectors of covariance matrix)


