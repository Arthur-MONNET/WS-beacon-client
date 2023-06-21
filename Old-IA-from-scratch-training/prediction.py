import numpy as np
from keras.models import load_model

# Chargement du modèle entraîné
model = load_model('saved_models/model.h5')

# Chargement des données de test prétraitées
X_test = np.load('data_preprocessed/features.npy')

# Prédictions
y_pred = model.predict(X_test)

# Affichage des prédictions
print(y_pred)