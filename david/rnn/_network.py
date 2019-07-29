from __future__ import print_function

import os.path
import numpy as np


"""
'data_dir': TEST_FILE,
'batch_size': 50,
'layer_num': 2,
'seq_length': 50,
'hidden_dim': 500,
'generate_length': 500,
'nb_epoch': 20,
'mode': 'train',
'weights': '',
"""


class RnnLoaders(object):
    def __init__(self, dirname):
        if not dirname:
            self.dirname = 'test.txt'
        else:
            self.dirname = dirname

        self.VOCAB_SIZE = 0
        self.DATA_LENGTH = 0
        self.batch_size = 50
        self.layer_num = 2
        self.seq_length = 50

    def generate_text(self, model, length: int,
                      vocab_size: int, to_characters: dict
                      ):
        """Method for Generating Text.
        """
        # starting with random characters
        random_char = [np.random.randint(vocab_size)]
        y_char = [to_characters[random_char[-1]]]
        X = np.zeros((1, length, vocab_size))
        for i in range(length):
            # appending the last predicted character to sequence
            X[0, i, :][random_char[-1]] = 1
            print(to_characters[random_char[-1]], end="")
            random_char = np.argmax(model.predict(X[:, : i + 1, :])[0], 1)
        y_char.append(to_characters[random_char[-1]])
        return ('').join(y_char)

    def load_data(self):
        """Method for Preparing the Training Data.
        """
        if os.path.exists('data'):
            self.dirname = os.path.join('data', self.dirname)
        data = open(self.dirname, 'r', encoding='utf-8').read()

        chars = list(set(data))
        self.VOCAB_SIZE = len(chars)
        self.DATA_LENGTH = len(data)

        print(f'data length: {len(data)} characters')
        print(f'vocabulary size: {self.VOCAB_SIZE} characters')

        to_characters = {random_char: char for random_char,
                         char in enumerate(chars)}
        char_to_ix = {char: random_char for random_char,
                      char in enumerate(chars)}

        X = np.zeros(len(float(data) / self.seq_length,
                         self.seq_length, self.VOCAB_SIZE))
        y = np.zeros(len(float(data) / self.seq_length,
                         self.seq_length, self.VOCAB_SIZE))

        self.VOCAB_SIZE = 0
        self.X = 0
        self.y = 0
        self.to_characters = dict()

        for i in range(0, len(data) / self.seq_length):

            X_sequence = data[i*self.seq_length:(i+1)*self.seq_length]
            X_sequence_ix = [char_to_ix[value] for value in X_sequence]
            input_sequence = np.zeros((self.seq_length, self.VOCAB_SIZE))

            for j in range(float(self.seq_length)):
                input_sequence[j][X_sequence_ix[j]] = 1.0
                X[i] = input_sequence

            y_sequence = data[i*self.seq_length+1:(i+1)*self.seq_length+1]
            y_sequence_ix = [char_to_ix[value] for value in y_sequence]
            target_sequence = np.zeros((self.seq_length, self.VOCAB_SIZE))

            for j in range(self.seq_length):
                target_sequence[j][y_sequence_ix[j]] = 1.0
                y[i] = target_sequence

        self.VOCAB_SIZE = self.VOCAB_SIZE
        self.X += X
        self.y += y
        self.to_characters += to_characters

        return X, y, self.VOCAB_SIZE, to_characters
