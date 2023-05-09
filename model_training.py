import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from data_preprocessing import prepare_dataset

# Define the number of classes
num_classes = 4

# Prepare the dataset
X, y = prepare_dataset()
print("Dataset shape: ", X.shape)

# Convert labels to categorical one-hot encoding
y = to_categorical(y, num_classes)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data to fit the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)

# Define the model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:], padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 4)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 4)))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

# Save the model
model.save('saved_models/model.h5')