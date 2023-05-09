# Sound Recognition AI
This project is a simple sound recognition AI system built using Keras and TensorFlow. It can recognize three different sounds: the hoot of an owl, a hunter's shot, and the chirping of a robin.

## Prerequisites
Before running the program, make sure you have Python `3.10` installed on your machine. You will also need to install the following libraries:

- numpy
- tensorflow
- pandas
- librosa
- keras
- pyaudio

You can install the required libraries by running the following command in your terminal:

```bash
pip install -r requirements.txt
``` 
## Getting Started
To start, you will need to have sound files of the three sounds you want to recognize. Place the sound files in the `data` directory under a subdirectory with the name of the sound. For example, if you want to recognize the hoot of an owl, create a subdirectory called `owl_hoot` in the `data` directory and place owl hoot sound files in it.

## Data Preprocessing
Before training the AI system, you need to preprocess the sound files and create the necessary data files. To do this, run the `data_preprocessing.py` file:

```bash
python data_preprocessing.py
```
The preprocessed data will be saved in the `data_preprocessed` directory.

## Model Training
Once the data preprocessing is complete, you can train the AI system. To do this, run the `model_training.py` file:

```bash
python model_training.py
```
The trained model will be saved in the `saved_models` directory.

## Real-time Sound Recognition
To recognize sounds in real-time using the microphone, run the `real_time_sound_recognition.py` file:

```bash
python real_time_sound_recognition.py
```
This will initialize the microphone and start recognizing sounds. The recognized sound class will be printed to the console.

## Testing the Model
To test the accuracy of the trained AI system, run the `model_testing.py` file:

```bash
python model_testing.py
```
This will evaluate the model on the test data and print a classification report.

## Making Predictions
You can also use the trained model to make predictions on new sound files. To do this, run the `prediction.py` file:

```bash
python prediction.py path/to/sound/file.wav
```
Replace `path/to/sound/file.wav` with the path to the sound file you want to predict the class of. The predicted sound class will be printed to the console.
