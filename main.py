import numpy as np
import matplotlib.pyplot as graph

from neurone import Neurone
from sklearn.datasets import make_blobs


def main():
    ## Datasets
    X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    Y = Y.reshape((Y.shape[0], 1))

    ## Programme
    n = Neurone(X, Y)
    pred = n._learn(1000)

    ## Graphe
    # Fonction coût
    graph.plot(n.cout)
    graph.title(str(pred*100) + "%")
    graph.show()

    graph.plot(n.predictions)
    graph.show()

    # Frontière de décision
    x0 = np.linspace(-2, 5, 100)
    x1 = (-n.W[0] * x0 - n.b) / n.W[1]

    graph.scatter(X[:, 0], X[:, 1], c=Y, cmap="summer")
    graph.plot(x0, x1, c='blue', lw=2)
    graph.show()


main()