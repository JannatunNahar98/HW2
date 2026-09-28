"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Question 2: Sentiment Classification Using RNN
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the IMDB dataset
VOCAB_SIZE = 10000
MAX_LEN = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

# 2. Tokenization is already represented as integer word IDs by keras IMDB.
# Pad/truncate every review to the same length.
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

print("Training shape:", x_train.shape)
print("Testing shape :", x_test.shape)

# 3. LSTM-based sentiment classifier
model = tf.keras.Sequential([
    layers.Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN),
    layers.LSTM(64),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Predictions
probabilities = model.predict(x_test, verbose=0).ravel()
y_pred = (probabilities >= 0.5).astype(int)

# 4. Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"],
    digits=4
))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)
disp.plot()
plt.title("IMDB Sentiment Confusion Matrix")
plt.tight_layout()
plt.savefig("q2_confusion_matrix.png", dpi=200)
plt.show()

# 5. Precision-recall tradeoff explanation
print("""
Precision-recall tradeoff:
Precision measures how many reviews predicted as positive are actually positive.
Recall measures how many of all truly positive reviews are correctly identified.
Changing the classification threshold changes this balance. A lower threshold
usually increases recall but can reduce precision, while a higher threshold
can increase precision but reduce recall. Both metrics matter because a model
that only favors one can miss important errors in the other direction.
""")

model.save("q2_imdb_lstm.keras")
