import pyaudio
import numpy as np
import tensorflow as tf
import librosa
import websocket
import json
from datetime import datetime, timedelta

# Load the trained model
model = tf.keras.models.load_model("saved_models/model_full.h5")

# Define the sampling rate and frame length
sr = 48000
frame_length = int(sr / 94 * 44)

# Initialize the microphone
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt32, channels=1, rate=sr, input=True, frames_per_buffer=frame_length)

# Initialize the WebSocket
ws = websocket.WebSocket()

def on_open():
    print('WebSocket connected')

    # Send a message to indicate that the connection is established
    message = {'type': 'connection', 'sender': 'beacon', 'payload': 'Sound recognition started'}
    ws.send(json.dumps(message))

def on_close():
    print('WebSocket disconnected')

ws.on_open = on_open
ws.on_close = on_close
ws.connect('wss://9b9c-2a01-cb16-202b-c55f-7058-255e-300c-800a.eu.ngrok.io')

# Start the real-time sound recognition
while True:
    # Read a frame of audio data
    data = stream.read(frame_length)

    # Convert the data to a numpy array
    samples = np.frombuffer(data, dtype=np.int32)

    # Convert the samples to float and normalize between -1 and 1
    samples = samples.astype('float32') / 2147483647.0

    # Extract features from the audio data
    features = librosa.feature.mfcc(samples, sr=44)

    # Reshape the feature array for the model input
    features = np.reshape(features, (1, features.shape[0], features.shape[1], 1))

    # Predict the sound class
    prediction = model.predict(features)

    # Get the predicted sound class
    sound_class = np.argmax(prediction, axis=1)

    if sound_class == 0:
        print("cerf")
        message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reporting': 1, 'location': 'montagne-de-vuache' }}
        ws.send(json.dumps(message))
    elif sound_class == 1:
        print("coup_de_feu_chasseur")
        message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reporting': 2, 'location': 'montagne-de-vuache' }}
        ws.send(json.dumps(message))
    elif sound_class == 2:
        print("Gelinotte des bois")
        message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reporting': 3, 'location': 'montagne-de-vuache' }}
        ws.send(json.dumps(message))
    elif sound_class == 3:
        print("Loups")
        message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reporting': 4, 'location': 'montagne-de-vuache' }}
        ws.send(json.dumps(message))
    elif sound_class == 4:
        print("moto_cross")
        message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reporting': 5, 'location': 'montagne-de-vuache' }}
        ws.send(json.dumps(message))
    elif sound_class == 5:
        print("Renard")
        message = {'type': 'new-alert', 'sender': 'beacon', 'payload': {'reporting': 6, 'location': 'montagne-de-vuache' }}
        ws.send(json.dumps(message))
    else:
        print("")