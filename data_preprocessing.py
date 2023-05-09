import os
import numpy as np
import librosa

# Définir les classes de sons à reconnaître
classes = ['_white', 'chouette_hulotte', 'hunter_shot', 'rouge_gorge']

# Définir les paramètres d'analyse des fichiers audio
sampling_rate = 22050
duration = 1  # réduire la durée pour traiter le son en temps réel
hop_length = 512
n_mels = 128
n_fft = 2048
n_mfcc = 20

def extract_features(signal):
    # Extraire les caractéristiques MFCC avec une longueur fixe
    mfccs = librosa.feature.mfcc(signal, sr=sampling_rate, n_fft=n_fft,
                                 hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)

    # Ajouter une dimension pour obtenir un tableau de 4 dimensions
    mfccs = np.expand_dims(mfccs, axis=-1)

    return mfccs

def prepare_dataset():
    X = []
    y = []

    # Parcourir les classes de sons à reconnaître
    for i, cls in enumerate(classes):
        # Extraire les fichiers sonores de la classe
        for file_name in os.listdir(os.path.join('data', cls)):
            file_path = os.path.join('data', cls, file_name)

            # Charger le fichier audio
            signal, sr = librosa.load(file_path, sr=sampling_rate, duration=duration)

            # Extraire les caractéristiques du fichier audio
            features = extract_features(signal)

            # Ajouter les caractéristiques et la classe correspondante aux listes
            X.append(features)
            y.append(i)

    # Convertir les listes en tableaux numpy
    X = np.array(X)
    y = np.array(y)

    return X, y

if __name__ == '__main__':
    # Prétraiter les données et les enregistrer dans des fichiers numpy
    X, y = prepare_dataset()
    np.save('data_preprocessed/features.npy', X)
    np.save('data_preprocessed/labels.npy', y)