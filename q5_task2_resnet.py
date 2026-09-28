"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Question 5 Task 2: Residual Block and Simple ResNet-like Model
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

def residual_block(input_tensor, filters=64):
    """
    Residual block with two 3x3 convolutions and a skip connection.
    The second convolution is linear before addition so that the skip
    connection is added before the final ReLU activation.
    """
    x = layers.Conv2D(
        filters, (3, 3), padding="same", activation="relu"
    )(input_tensor)

    x = layers.Conv2D(
        filters, (3, 3), padding="same", activation=None
    )(x)

    x = layers.Add()([input_tensor, x])
    x = layers.ReLU()(x)
    return x

# Input shape is chosen as a standard image size.
inputs = layers.Input(shape=(224, 224, 3))

x = layers.Conv2D(
    64, (7, 7), strides=2, padding="same", activation="relu"
)(inputs)

x = residual_block(x, 64)
x = residual_block(x, 64)

x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)

model = Model(inputs, outputs, name="Simple_ResNet")

print("\nResNet-like Model Summary:")
model.summary()
