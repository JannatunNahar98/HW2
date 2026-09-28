"""
CS5720 Neural Network and Deep Learning - Home Assignment 2
Question 1: Implementing an RNN for Text Generation

This script downloads the Shakespeare character dataset, trains an LSTM
to predict the next character, and generates text using temperature sampling.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load a text dataset
path = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)
text = open(path, "rb").read().decode(encoding="utf-8")

print("Text length:", len(text))

# 2. Convert characters to integer IDs
vocab = sorted(set(text))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text], dtype=np.int32)

# Create training sequences
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(sequence):
    input_seq = sequence[:-1]
    target_seq = sequence[1:]
    return input_seq, target_seq

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# 3. Define an LSTM-based RNN
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 512

model = tf.keras.Sequential([
    layers.Embedding(vocab_size, embedding_dim),
    layers.LSTM(rnn_units, return_sequences=True),
    layers.Dense(vocab_size)
])

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss)

model.summary()

# 4. Train the model
# Increase EPOCHS for a stronger model if your machine has enough time.
EPOCHS = 5
history = model.fit(dataset, epochs=EPOCHS)

# Save the trained model
model.save("q1_text_generation.keras")

def generate_text(model, start_string, num_generate=500, temperature=1.0):
    """Generate characters one at a time using temperature scaling."""
    input_ids = [char2idx.get(c, 0) for c in start_string]
    input_tensor = tf.expand_dims(input_ids, 0)

    generated = []

    for _ in range(num_generate):
        predictions = model(input_tensor)
        predictions = predictions[:, -1, :] / temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        generated.append(idx2char[predicted_id])

        input_tensor = tf.concat(
            [input_tensor, tf.expand_dims([predicted_id], 0)], axis=1
        )

        # Keep the context from becoming unnecessarily large.
        if input_tensor.shape[1] > seq_length:
            input_tensor = input_tensor[:, -seq_length:]

    return start_string + "".join(generated)

print("\nGenerated text (temperature = 0.8):")
print(generate_text(model, "ROMEO:", 300, temperature=0.8))

print("\nGenerated text (temperature = 1.5):")
print(generate_text(model, "ROMEO:", 300, temperature=1.5))

print("""
Temperature explanation:
- Lower temperature (< 1) makes the probability distribution sharper,
  so generation is more predictable and conservative.
- Temperature = 1 uses the original predicted distribution.
- Higher temperature (> 1) makes the distribution flatter, increasing
  randomness and producing more varied, but potentially less coherent, text.
""")
