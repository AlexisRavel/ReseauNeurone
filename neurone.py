import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Neurone:
    ## Variables
    # Données
    X: np.ndarray
    Y: np.ndarray
    Xt = 0
    Yt = 0

    # Paramètres
    W: np.ndarray
    b: np.ndarray

    # Travail neurone
    learningRate: float
    A: np.ndarray
    cout = [] # Stocker l'évolution du cout du modele
    coutTest = [] # Stocker l'évolution du cout du modele avec les données de test
    predictions = [] # Stocker l'évolution de la vraisemblance du modèle
    predictionsTest = [] # Stocker l'évolution de la vraisemblance du modèle avec les données de test

    ## Fonctions
    # Entrée: matX, matY --> nos données d'entrainements
    #         testX, testY --> des données de test pour diagnostiquer le surapprentissage
    #         learningRate --> Pas d'apprentissage
    # Traitement: Init les variables de données X et Y ainsi que les paramètres W et b
    # Sortie: Rien
    def __init__(self, matX:np.ndarray, matY:np.ndarray, testX:np.ndarray=None, testY:np.ndarray=None, lR=0.2):
        self.X = matX
        self.Y = matY
        if type(testX)==np.ndarray and type(testY)==np.ndarray:
            self.Xt = testX
            self.Yt = testY
        self.learningRate = lR

        self.W = np.random.randn(self.X.shape[1], 1)
        self.b = np.random.randn(1)
    
    # Modele linéaire #
    # Entrée: X les données à traiter sur le modèle
    # Traitement: Calcul de Z et de A, le modèle
    # Sortie: A
    def _calcul_modele(self, X):
        # Z = X * W + b
        Z = X.dot(self.W) + self.b
        # A = 1 / 1 + e^-Z
        return 1/(1+np.exp(-Z))

    # Entrée: Rien
    # Traitement: Calcul les gradients afin de pouvoir ajuster les param
    # Sortie: Le tuple de gradient (dW, db)
    def _gradients(self):
        dW = (1/self.Y.size)*np.dot(self.X.T, self.A-self.Y)
        db = (1/self.Y.size)*np.sum(self.A-self.Y)
        return (dW, db)

    # Entrée: Rien 
    # Traitement: Ajuste les param en fonction des gradients
    # Sortie: Rien
    def _update_param(self):
        gradients = self._gradients()
        self.W = self.W - self.learningRate * gradients[0]
        self.b = self.b - self.learningRate * gradients[1]

    # Log Loss #
    # Entrée: A --> la probabilité sur laquelle on calcule le coût
    # Traitement: Calcul le coût du modèle A calculer préalablement
    # Sortie: L --> Le coût calculer
    def _cout(self, A, Y):
        # Epsilon --> éviter d'avoir un 0 dans le log
        epsilon = 1e-15
        return -(1/Y.size)*(np.sum(Y*np.log(A+epsilon)+(1-Y)*np.log(1-A+epsilon)))

    # Entrée: Le nombre d'itération
    # Traitement: Update les paramètres et calcul le cout du modele
    # Sortie: La vraisemblance du modèle à la fin de l'apprentissage
    def _learn(self, iter):
        for i in tqdm(range(iter)):
            self.A = self._calcul_modele(self.X)
            self._update_param()
            self.cout.append(self._cout(self.A, self.Y))
            self.predictions.append(accuracy_score(self.Y, self._predict(self.X)))
            # On calcul aussi le cout et la vraisemblance avec les données test si on en a
            if type(self.Xt)==np.ndarray:
                self.coutTest.append(self._cout(self._calcul_modele(self.Xt), self.Yt))
                self.predictionsTest.append(accuracy_score(self.Yt, self._predict(self.Xt)))
        # Le résultat de l'apprentissage
        predOfY = self._predict(self.X)
        return accuracy_score(self.Y, predOfY)
    
    # Entrée: Les données X à prédire
    # Traitement: Prédiciton des données
    # Sortie: Prédiction des données
    def _predict(self, X):
        A = self._calcul_modele(X)
        return A >= 0.5