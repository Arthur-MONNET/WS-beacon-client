import pyaudio
import numpy as np
import tensorflow as tf
import librosa

# Load the trained model
model = tf.keras.models.load_model("saved_models/model.h5")

# Define the sampling rate and frame length
sr = 22050
frame_length = sr

# Initialize the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=frame_length)

# Start the real-time sound recognition
while True:
    # Read a frame of audio data
    data = stream.read(frame_length)
    # Convert the data to a numpy array
    samples = np.frombuffer(data, dtype=np.int16)
    # Convert the samples to float and normalize between -1 and 1
    samples = samples.astype('float32') / 32767.0
    # Extract features from the audio data
    features = librosa.feature.mfcc(samples, sr=sr)
    # Reshape the feature array for the model input
    features = np.reshape(features, (1, features.shape[0], features.shape[1], 1))
    # Predict the sound class
    prediction = model.predict(features)
    # Get the predicted sound class
    sound_class = np.argmax(prediction, axis=1)
    # Print the predicted sound class en francais
    if(sound_class == 0):
        print("")
    elif(sound_class == 1):
        print("Chouette hulotte")
    elif(sound_class == 2):
        print("Tir de chasse")
    elif(sound_class == 3):
        print("Rouge-gorge")