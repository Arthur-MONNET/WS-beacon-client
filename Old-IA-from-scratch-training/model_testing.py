import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

# Chargement des données de test prétraitées
X_test = np.load('data_preprocessed/features.npy')
y_test = np.load('data_preprocessed/labels.npy')

model = load_model('saved_models/model_full.h5')

y_pred = np.argmax(model.predict(X_test), axis=1)

target_names = ['cerf', 'coup_de_feu_chasseur', 'Gelinotte des bois', 'Loups', 'moto_cross', 'Renard', 'white']
print(classification_report(y_test, y_pred, target_names=target_names))
