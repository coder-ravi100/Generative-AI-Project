import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import string
import os

# Load dataset
with open("../data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Remove punctuation
text = text.translate(str.maketrans("", "", string.punctuation))

# Character vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Create sequences
seq_length = 40
X, y = [], []

for i in range(len(text) - seq_length):
    X.append([char_to_idx[c] for c in text[i:i + seq_length]])
    y.append(char_to_idx[text[i + seq_length]])

X = np.array(X)
y = tf.keras.utils.to_categorical(y, vocab_size)

# Build model
model = Sequential([
    Embedding(vocab_size, 64, input_length=seq_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(vocab_size, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train
model.fit(X, y, epochs=15, batch_size=128)

# Save model
os.makedirs("../outputs", exist_ok=True)
model.save("../outputs/lstm_text_model.h5")

print("Model training complete & saved.")
