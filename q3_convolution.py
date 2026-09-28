"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Question 3: Convolution Operations with Different Parameters
"""

import numpy as np
import tensorflow as tf

input_matrix = np.array([
    [1,  2,  3,  4,  5],
    [6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

# TensorFlow Conv2D performs cross-correlation, which is the convention
# used by Keras convolution layers.
x = input_matrix.reshape(1, 5, 5, 1)
k = kernel.reshape(3, 3, 1, 1)

def run_conv(stride, padding):
    output = tf.nn.conv2d(
        x,
        k,
        strides=[1, stride, stride, 1],
        padding=padding
    )
    return output.numpy().squeeze()

cases = [
    (1, "VALID"),
    (1, "SAME"),
    (2, "VALID"),
    (2, "SAME")
]

print("Input matrix:\n", input_matrix)
print("\nKernel:\n", kernel)

for stride, padding in cases:
    result = run_conv(stride, padding)
    print(f"\nStride = {stride}, Padding = '{padding}'")
    print(result)

print("""
Expected feature maps for this input/kernel:

Stride = 1, Padding = 'VALID'
[[0. 0. 0.]
 [0. 0. 0.]
 [0. 0. 0.]]

Stride = 1, Padding = 'SAME'
[[  4.   3.   2.   1.  -6.]
 [ -5.   0.   0.   0. -11.]
 [-10.   0.   0.   0. -16.]
 [-15.   0.   0.   0. -21.]
 [-46. -27. -28. -29. -56.]]

Stride = 2, Padding = 'VALID'
[[0. 0.]
 [0. 0.]]

Stride = 2, Padding = 'SAME'
[[  4.   2.  -6.]
 [-10.   0. -16.]
 [-46. -28. -56.]]
""")
