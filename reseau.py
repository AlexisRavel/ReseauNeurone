import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss


class Reseau:
    ## Variables
    # Données
    X: np.ndarray # Nos données d'entrée
    Y: np.ndarray # Les valeurs associées aux données d'entrée

    # Paramètres
    param = {}
    nbC: int # Le nombre de couches du réseau

    # Travail du réseau
    learningRate: float
    A = {}
    app: np.ndarray


    ## Fonctions
    # Entrée: format --> les dimensions du réseau
    #         matX et matY --> les matrices de données d'entrée
    #         lR --> le pas d'apprentissage
    # Traitement: Init les paramètres en fonction des données
    # Sortie: Rien
    def __init__(self, format, matX, matY, lR=0.2):
        self.X = matX
        self.Y = matY
        self.nbC = len(format)
        self.learningRate = lR

        for c in range(1, self.nbC):
            self.param["W" + str(c)] = np.random.randn(format[c], format[c-1])
            self.param["b" + str(c)] = np.random.randn(format[c], 1)
    
    # Entrée: X --> les données à faire vérifier par le modèle
    # Traitement: calcul de Z puis de A
    # Sortie: A
    def _forward_propagation(self, X):
        # Z1 = W1 * X + b
        # Zn = Wn * An-1 + b
        # A = 1/1+e-z
        A = {"A0": X}
        for c in range(1, self.nbC):
            Z = self.param["W" + str(c)].dot(A["A" + str(c-1)]) + self.param["b" + str(c)]
            A["A" + str(c)] = 1 / (1+np.exp(-Z))
        return A
    
    # Entrée: Rien
    # Traitement: calcul les gradients pour ajuster le modèle
    # Sortie: gradients
    def _gradients(self):
        dZ = self.A["A" + str(self.nbC-1)] - self.Y
        gradients = {}
        for c in reversed(range(1, self.nbC)):
            gradients["dW" + str(c)] = 1/self.Y.size * np.dot(dZ, self.A["A" + str(c-1)].T)
            gradients["db" + str(c)] = 1/self.Y.size * np.sum(dZ)
            if c > 1:
                dZ = np.dot(self.param["W" + str(c)].T, dZ) * self.A["A" + str(c-1)] * (1 - self.A["A" + str(c-1)])
        return gradients

    # Entrée: Rien
    # Traitement: Ajuster les paramètres du réseau
    # Sortie: Rien
    def _back_propagation(self):
        gradients = self._gradients()
        for c in range(1, self.nbC):
            self.param["W" + str(c)] = self.param["W" + str(c)] - self.learningRate * gradients["dW" + str(c)]
            self.param["b" + str(c)] = self.param["b" + str(c)] - self.learningRate * gradients["db" + str(c)]

    # Entrée: iter --> le nombre d'itération pour l'apprentissage du réseau
    # Traitement: la forward et la back propagation
    # Sortie: predOfY
    def _learn(self, iter):
        self.app = np.zeros((iter, 2))

        for i in tqdm(range(iter)):
            self.A = self._forward_propagation(self.X)
            self._back_propagation()
            # Pour la visualisation
            lastA = self.A["A" + str(self.nbC-1)]
            self.app[i, 0] = (log_loss(self.Y.flatten(), lastA.flatten()))
            predOfY = self._predict(self.X)
            self.app[i, 1] = (accuracy_score(self.Y.flatten(), predOfY.flatten()))
        return accuracy_score(self.Y, predOfY)

    # Entrée: X --> les données sur lesquelles appliquées le modèle
    # Traitement: forward propagation
    # Sortie: la prediction
    def _predict(self, X):
        A = self._forward_propagation(X)
        return A["A" + str(self.nbC-1)] >= 0.5