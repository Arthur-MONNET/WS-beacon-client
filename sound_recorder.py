import sounddevice as sd
import scipy.io.wavfile as wav

# Define the sampling rate and duration of the recording
sr = 48000
duration = 300  # Recording duration in seconds

# Start the recording
print("Recording started...")
recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
sd.wait()  # Wait for the recording to complete

# Save the recording to a WAV file
wav.write('noise_8.wav', sr, recording)

print("Recording saved as recording.wav")
