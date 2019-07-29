from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed

from .utils import generate_text
from .utils import load_data

# parameters for network definition
from .config import network as _network
NET = _network()


X, y, VOCAB_SIZE, to_characters = load_data(NET.data_dir, NET.seq_length)


# creating and compiling the network
model = Sequential()
model.add(LSTM(NET.hidden_dim, input_shape=(
    None, VOCAB_SIZE), return_sequences=True))

for i in range(NET.layer_num - 1):
    model.add(LSTM(NET.hidden_dim, return_sequences=True))


model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))

# generate a sample before training.
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
generate_text(model, NET.generate_length, VOCAB_SIZE, to_characters)

if not NET.weights == '':
    model.load_weights(NET.weights)
    nb_epoch = int(NET.weights[NET.weights.rfind(
        '_') + 1: NET.weights.find('.')])
else:
    nb_epoch = 0

# training if there is no trained weights specified
if NET.mode == 'train' or NET.weights == '':
    while True:
        print(f'\n\n::epoch@ {nb_epoch}\n')
        model.fit(X, y, batch_size=NET.batch_size, verbose=1, nb_epoch=1)
        nb_epoch += 1
        generate_text(model, NET.generate_length, VOCAB_SIZE, to_characters)
        if nb_epoch % 10 == 0:
            model.save_weights(
                'checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(
                    NET.layer_num, NET.hidden_dim, nb_epoch)
            )

        # else, loading the trained weights and performing
        # generation only loading the trained weights

elif NET.weights == '':
    model.load_weights(NET.weights)
    generate_text(model, NET.generate_length, VOCAB_SIZE, to_characters)
    print('\n\n')
else:
    print('\n\nEmpty')
