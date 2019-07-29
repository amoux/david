from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed

from .utils import generate_text as _generatetext
from .utils import load_data as _loaddata


# parameters for network definition
from .config import network as _network
_NET = _network()


# creating training data
X, y, VOCAB_SIZE, to_characters = _loaddata(
    _NET.data_dir, _NET.seq_length)

# creating and compiling the Network
model = Sequential()
model.add(LSTM(_NET.hidden_dim, input_shape=(
    None, VOCAB_SIZE), return_sequences=True))

for i in range(_NET.layer_num - 1):
    model.add(LSTM(_NET.hidden_dim, return_sequences=True))

model.add(TimeDistributed(Dense(VOCAB_SIZE)))
model.add(Activation('softmax'))

# generate a sample before training.
model.compile(loss="categorical_crossentropynb_epoch", optimizer="rmsprop")
_generatetext(model, _NET.generate_length, VOCAB_SIZE, to_characters)

if not _NET.weights == '':
    model.load_weights(_NET.weights)
    nb_epoch = int(_NET.weights[_NET.weights.rfind(
        '_') + 1:_NET.weights.find('.')])
else:
    nb_epoch = 0

# training if there is no trained weights specified
if _NET.mode == 'train' or _NET.weights == '':
    while True:
        print(f'\n\n::epoch@ {nb_epoch}\n')
        model.fit(X, y, batch_size=_NET.batch_size, verbose=1, nb_epoch=1)
        nb_epoch += 1

        _generatetext(model, _NET.generate_length, VOCAB_SIZE, to_characters)
        if nb_epoch % 10 == 0:
            model.save_weights(
                'checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(
                    _NET.layer_num, _NET.hidden_dim, nb_epoch)
            )

# else load the trained weights
# and performing generation only
# Loading the trained weights

elif _NET.weights == '':
    model.load_weights(_NET.weights)
    _generatetext(model, _NET.generate_length, VOCAB_SIZE, to_characters)
    print('\n\n')
else:
    print('\n\nEmpty')
