import numpy as np

from sklearn.metrics import accuracy_score


class Neurone:
    ## Variables ##
    # Données
    # X:ndarray la matrice des données
    # Y:ndarray la matrice des "étiquettes" des données correspondantes (0 ou 1)
    
    # Paramètres
    W = None # la matrice des paramètres de poids
    b = 0 # le paramètre du biais
    learningRate = 0 # le pas positif pour mettre l'avancer du programme d'apprentissage du neurone

    # Résultat des fonctions
    A = None # la matrice du modèle0
    cout = [] # le cout stocker à chaque itération

    ## Fonctions ##
    # Entrée: la matrice X des données et sa matrice Y associé
    # Traitement: Initialise les paramètres en fonction des données
    # Sortie: Rien   
    def __init__(self, matX, matY, lR=0.2):
        # init des données
        self.X = matX
        self.Y = matY
        self.learningRate = lR

        # peuplement des paramètres
        self.W = np.random.randn(self.X.shape[1], 1)
        self.b = np.random.randn(1)

    # Entrée: la matrice X de données sur laquelle définir le modèle
    # Traitement: Calculer la "position" de la donnée par rapport à la frontière de décision Z et la probabilité A de cette dernière
    # Sortie: A   
    def _linear_model(self, X):
        # Z = X*W + b
        Z = X.dot(self.W) + self.b
        # A = 1/1+e^(-Z)
        return 1/(1+np.exp(-Z))

    # Entrée: Rien
    # Traitement: Calcul le cout (le taux d'erreur) du modèle
    # Sortie: Rien   
    def _update_cost(self):
        A = self._linear_model(self.X)
        self.cout.append((-1/self.Y.size)*(np.sum(self.Y*np.log(A)+(1-self.Y)*np.log(1-A))))

    # Entrée: Rien
    # Traitement: Calculer les dérivées partielles afin de dterminer l'évolution de la courbe et donc des paramètres
    # Sortie: dW et db les valeurs des dérivées partielles des paramètres W et b
    def _gradient(self):
        A = self._linear_model(self.X)
        dW = (1/self.Y.size)*np.dot(self.X.T, A-self.Y)
        db = (1/self.Y.size)*np.sum(A-self.Y)
        return (dW, db)
    
    # Entrée: Rien
    # Traitement: Mettre à jour les paramètres W et b en fonction des gradients
    # Sortie: Rien  
    def _update(self):
        self._update_cost()
        gradients = self._gradient()
        self.W = self.W - self.learningRate * gradients[0]
        self.b = self.b - self.learningRate * gradients[1]

    # Entrée: Nombre d'itération
    # Traitement: Selon le nombre d'itération voulue i, fait tourner update() xi, plsu calcul la prediction du modèle
    # Sortie: predOfY la vraisemblance de notre modèle  
    def _learn(self, iteration):
        for i in range(iteration):
            self._update()
        predOfY = self._predict(self.X)
        return(accuracy_score(self.Y, predOfY))

    # Entrée: La donnée X à traiter, si X=none, on prends alors la matrice X de données du neurone qui a été init
    # Traitement: Selon le nombre d'itération voulue i, fait tourner update() xi
    # Sortie: Rien  
    def _predict(self, X):
        A = self._linear_model(X)
        return A >= 0.5