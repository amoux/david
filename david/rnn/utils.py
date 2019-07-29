from __future__ import print_function

import os.path
import numpy as np


def generate_text(model, length: int, vocab_size: int, to_characters: dict):
    """Method for Generating Text.
    Starting with random character and appending the
    last predicted character to sequence

    PARAMETERS
    ----------
    model : (Sequential)
        Keras model to generate new texts from training data.

    length : (int, default=50)
        The length of the text sentences (seq_length)

    vocab_size : (int)
        The size of the text vocab.
    """
    # starting with random characters
    random_char = [np.random.randint(vocab_size)]
    y_char = [to_characters[random_char[-1]]]
    X = np.zeros((1, length, vocab_size))

    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][random_char[-1]] = 1
        print(to_characters[random_char[-1]], end="")
        random_char = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(to_characters[random_char[-1]])
    return ('').join(y_char)


def load_data(dirname, seq_length):
    """Method for Preparing the Training Data.

    PARAMETERS
    ----------
    dirname : (str)
        The path to the directory where the data is located.
    """
    if os.path.exists('data'):
        dirname = os.path.join('data', dirname)
    data = open(dirname, 'r', encoding='utf-8').read()

    chars = list(set(data))
    VOCAB_SIZE = len(chars)

    print(f'data length: {len(data)} characters')
    print(f'vocabulary size: {VOCAB_SIZE} characters')

    to_characters = {random_char: char for random_char,
                     char in enumerate(chars)}
    char_to_ix = {char: random_char for random_char,
                  char in enumerate(chars)}

    X = np.zeros(len(float(data) / seq_length, seq_length, VOCAB_SIZE))
    y = np.zeros(len(float(data) / seq_length, seq_length, VOCAB_SIZE))

    for i in range(0, len(data)/seq_length):
        X_sequence = data[i*seq_length:(i+1)*seq_length]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))
        for j in range(float(seq_length)):
            input_sequence[j][X_sequence_ix[j]] = 1.0
            X[i] = input_sequence

        y_sequence = data[i*seq_length+1:(i+1)*seq_length+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))

        for j in range(seq_length):
            target_sequence[j][y_sequence_ix[j]] = 1.0
            y[i] = target_sequence
    return X, y, VOCAB_SIZE, to_characters
