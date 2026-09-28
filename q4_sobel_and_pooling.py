"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Question 4: CNN Feature Extraction with Filters and Pooling
"""

from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------- Task 1: Sobel Edge Detection ----------------
IMAGE_PATH = Path("sample_image.png")

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(
        f"Could not load {IMAGE_PATH}. Keep sample_image.png in the same folder."
    )

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=np.float32)

edge_x = cv2.filter2D(image, cv2.CV_32F, sobel_x)
edge_y = cv2.filter2D(image, cv2.CV_32F, sobel_y)

# Convert signed responses to displayable images
edge_x_display = cv2.convertScaleAbs(edge_x)
edge_y_display = cv2.convertScaleAbs(edge_y)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(edge_x_display, cmap="gray")
plt.title("Sobel-X")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(edge_y_display, cmap="gray")
plt.title("Sobel-Y")
plt.axis("off")

plt.tight_layout()
plt.savefig("q4_sobel_results.png", dpi=200)
plt.show()

# ---------------- Task 2: Max and Average Pooling ----------------
np.random.seed(42)

matrix = np.random.randint(0, 10, size=(4, 4)).astype(np.float32)
input_tensor = matrix.reshape(1, 4, 4, 1)

max_pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
)
avg_pool = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
)

max_result = max_pool(input_tensor).numpy().squeeze()
avg_result = avg_pool(input_tensor).numpy().squeeze()

print("\nOriginal 4x4 matrix:")
print(matrix)

print("\n2x2 Max Pooled matrix:")
print(max_result)

print("\n2x2 Average Pooled matrix:")
print(avg_result)
