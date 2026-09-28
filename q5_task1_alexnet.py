"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Question 5 Task 1: Simplified AlexNet
"""

import tensorflow as tf
from tensorflow.keras import layers, models

INPUT_SHAPE = (224, 224, 3)

model = models.Sequential([
    layers.Input(shape=INPUT_SHAPE),

    layers.Conv2D(96, (11, 11), strides=4, activation="relu"),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    layers.Conv2D(256, (5, 5), activation="relu"),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    layers.Conv2D(384, (3, 3), activation="relu"),
    layers.Conv2D(384, (3, 3), activation="relu"),
    layers.Conv2D(256, (3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(3, 3), strides=2),

    layers.Flatten(),
    layers.Dense(4096, activation="relu"),
    layers.Dropout(0.50),
    layers.Dense(4096, activation="relu"),
    layers.Dropout(0.50),
    layers.Dense(10, activation="softmax")
], name="Simplified_AlexNet")

print("\nSimplified AlexNet Model Summary:")
model.summary()
