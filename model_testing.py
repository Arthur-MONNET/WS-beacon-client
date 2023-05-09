import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

# Chargement des données de test prétraitées
X_test = np.load('data_preprocessed/features.npy')
y_test = np.load('data_preprocessed/labels.npy')

# Chargement du modèle entraîné
model = load_model('saved_models/model.h5')

# Evaluation du modèle sur les données de test
y_pred = np.argmax(model.predict(X_test), axis=1)

# Affichage du rapport de classification
target_names = ['chouette_hulotte', 'hunter_shot', 'rouge_gorge']
print(classification_report(y_test, y_pred, target_names=target_names))
