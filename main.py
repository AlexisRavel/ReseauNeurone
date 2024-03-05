import numpy as np
import matplotlib.pyplot as plt

from neurone import Neurone
from sklearn.datasets import make_blobs


## Main programme
# Généré les données de base, le dataSet
X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
Y = Y.reshape((Y.shape[0], 1))

# Neurone
n = Neurone(X, Y)
premierPred = n._learn(100)
print(premierPred)

plt.plot(n.cout)
plt.show()

newData = np.array([2, 1])
print(n._predict(newData))

x0 = np.linspace(-2, 5, 100)
x1 = (-n.W[0] * x0 - n.b) / n.W[1]

plt.scatter(X[:, 0], X[:, 1], c=Y, cmap="summer")
plt.scatter(newData[0], newData[1], c='r')
plt.plot(x0, x1, c='orange', lw=3)
plt.show()