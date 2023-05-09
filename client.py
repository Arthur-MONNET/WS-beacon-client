import pyaudio
import numpy as np
import tensorflow as tf
import librosa
import websocket
import json

# Load the trained model
model = tf.keras.models.load_model("saved_models/model.h5")

# Define the sampling rate and frame length
sr = 22050
frame_length = sr

# Initialize the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sr, input=True, frames_per_buffer=frame_length)

# Initialize the WebSocket
ws = websocket.WebSocket()

def on_open():
    print('WebSocket connected')

    # Send a message to indicate that the connection is established
    message = {'type': 'connection', 'payload': 'Sound recognition started'}
    ws.send(json.dumps(message))

def on_close():
    print('WebSocket disconnected')

ws.on_open = on_open
ws.on_close = on_close
ws.connect('ws://192.168.1.15:8080')

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
        message = {'type': 'balise', 'payload': 'Chouette hulotte'}
        ws.send(json.dumps(message))
    elif(sound_class == 2):
        print("Tir de chasse")
        message = {'type': 'balise', 'payload': 'Tir de chasse'}
        ws.send(json.dumps(message))
    elif(sound_class == 3):
        print("Rouge-gorge")
        message = {'type': 'balise', 'payload': 'Rouge-gorge'}
        ws.send(json.dumps(message))