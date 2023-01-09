# Import required libraries
import tensorflow as tf
import numpy as np
import os
import pickle
import tqdm
from string import punctuation

# # Import text
# import requests
# content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
# open("wonderland.txt", "w", encoding="utf-8").write(content)

sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 30
# dataset file path
FILE_PATH = "wonderland.txt"
BASENAME = os.path.basename(FILE_PATH)
# read the data
text = open(FILE_PATH, encoding="utf-8").read()
# remove caps, comment this code if you want uppercase characters as well
text = text.lower()
# remove punctuation
text = text.translate(str.maketrans("", "", punctuation))
# Get all unique characters in the text
vocab = ''.join(sorted(set(text)))
n_unique_chars = len(vocab)
# Since we have vocab as our vocabulary that contains all the unique characters of our dataset,
# we can make two dictionaries that map each character to an integer number and vice-versa:
# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(vocab)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(vocab)}
# convert all text into integers
encoded_text = np.array([char2int[c] for c in text])
# construct tf.data.Dataset object
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
# build sequences by batching
sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)


def split_sample(sample):
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        # extend the dataset with these samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds


def one_hot_samples(input_, target):
    # onehot encode the inputs and the targets
    # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)


# prepare inputs and targets
dataset = sequences.flat_map(split_sample)
dataset = dataset.map(one_hot_samples)
# repeat, shuffle and batch the dataset
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dense(n_unique_chars, activation="softmax"),
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
model_weights_path = f"results/{BASENAME}-{sequence_length}.h5"
model.save(model_weights_path)

# Generating new text
# We need a sample text to start generating.
seed = "chapter xiii"
vocab_size = len(char2int)
# generate 400 characters
s = seed
n_chars = 400
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text"):
    # make the input sequence
    X = np.zeros((1, sequence_length, vocab_size))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    # predict the next character
    predicted = model.predict(X, verbose=0)[0]
    # converting the vector to an integer
    next_index = np.argmax(predicted)
    # converting the integer to a character
    next_char = int2char[next_index]
    # add the character to results
    generated += next_char
    # shift seed and the predicted character
    seed = seed[1:] + next_char

print("Seed:", s)
print("Generated text:")
print(generated)

