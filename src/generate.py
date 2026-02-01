import numpy as np
import tensorflow as tf
import string

# Load model
model = tf.keras.models.load_model("../outputs/lstm_text_model.h5")

# Load text
with open("../data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

text = text.translate(str.maketrans("", "", string.punctuation))

chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 40

def generate_text(seed, length=300):
    result = seed
    for _ in range(length):
        seq = [char_to_idx[c] for c in result[-seq_length:]]
        seq = np.reshape(seq, (1, seq_length))
        prediction = model.predict(seq, verbose=0)
        next_char = idx_to_char[np.argmax(prediction)]
        result += next_char
    return result

seed_text = text[100:140]
generated = generate_text(seed_text)

with open("../outputs/sample_output.txt", "w") as f:
    f.write(generated)

print("Text generated & saved to outputs/sample_output.txt")
