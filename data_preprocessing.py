import os
import numpy as np
import librosa

# Définir les classes de sons à reconnaître
classes = ['_white', 'cerf', 'coup_de_feu_chasseur', 'Gelinotte des bois', 'Loups', 'moto_cross', 'Renard']

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

    max_length = 44  # Longueur maximale des caractéristiques

    for i, cls in enumerate(classes):
        for file_name in os.listdir(os.path.join('data', cls)):
            file_path = os.path.join('data', cls, file_name)
            signal, sr = librosa.load(file_path, sr=sampling_rate, duration=duration)
            features = extract_features(signal)

            # Ajuster les caractéristiques à la longueur maximale
            if features.shape[1] < max_length:
                features = np.pad(features, ((0, 0), (0, max_length - features.shape[1]), (0, 0)), mode='constant')
            elif features.shape[1] > max_length:
                features = features[:, :max_length, :]

            X.append(features)
            y.append(i)

    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == '__main__':
    # Prétraiter les données et les enregistrer dans des fichiers numpy
    X, y = prepare_dataset()
    np.save('data_preprocessed/features.npy', X)
    np.save('data_preprocessed/labels.npy', y)