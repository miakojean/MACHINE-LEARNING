import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print ('dimension de X:', X.shape)
print ('dimension de y:', y.shape)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.show()

def initialisation(X):
    W = np.random.randn(x.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)