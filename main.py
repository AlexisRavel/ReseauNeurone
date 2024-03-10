import numpy as np
import matplotlib.pyplot as graph

from neurone import Neurone
from reseau import Reseau
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles


def _neurone_main():
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


def _reseau_main():
    ## Datasets
    X, Y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    Y = Y.reshape((1, Y.shape[0]))

    ## Programme
    format = [X.shape[0], 16, 16, 16, 1]
    r = Reseau(format, X, Y)
    pred = r._learn(1000)
    print(pred)

    # Frontière de décision

    # Plot courbe d'apprentissage
    graph.figure(figsize=(12, 4))
    graph.subplot(1, 2, 1)
    graph.plot(r.app[:, 0], label='train loss')
    graph.legend()
    graph.subplot(1, 2, 2)
    graph.plot(r.app[:, 1], label='train acc')
    graph.legend()
    graph.show()


_reseau_main()