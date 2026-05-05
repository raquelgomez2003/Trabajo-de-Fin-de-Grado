import numpy as np
from sklearn.ensemble import RandomForestClassifier

def extraer_features(senales, fs):
    feats = []
    for signal in senales.values():
        feats.append(np.mean(signal))
        feats.append(np.std(signal))
    return np.array(feats)

class ModeloEstres:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def entrenar(self, X, y):
        self.model.fit(X, y)

    def predecir(self, senales, fs):
        feats = extraer_features(senales, fs).reshape(1, -1)
        return self.model.predict(feats)[0]