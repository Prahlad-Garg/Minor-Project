import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn(input_shape=(224, 224, 1)):

    model = models.Sequential()
    
    # Layer 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2), strides=2))    # Output: 112x112
    
    # Layer 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))    # Output: 56x56
    
    # Layer 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2), strides=2))    # Output: 28x28
    
    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    
    # 1 output node for Binary Classification (Pneumonia vs Normal), Using linear activation here since we can use BinaryCrossentropy in training
    model.add(layers.Dense(1, activation='linear'))
    
    return model
