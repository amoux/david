
# TEXT GENERATOR RNN

## Generates new texts from raw texts

TODO

* [ X ] Fix imports: `Got rid of unnecessary imports.`
* [ ] Optimize data control
* [ ] wrap tensorflow function in a method
* [ X ] I moved configurations, parameters, to the config.py
file. Now its easier and more organized to tune models settings.

* imports that were found not unused.

Unused modules:

    - keras.core.Dropout

    - keras.recurrent.SimpleRNN

> OLD ARGUMENT PARSERS (from train.py file)

```
('-data_dir', default='./data/test.txt')
('-batch_size', type=int, default=50)
('-layer_num', type=int, default=2)
('-seq_length', type=int, default=50)
('-hidden_dim', type=int, default=500)
('-generate_length', type=int, default=500)
('-nb_epoch', type=int, default=20)
('-mode', default='train')
('-weights', default='')

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']
```

> Got rid of parsing type arguments!

* These settings where on the recurrent_keras.py file,
which I found to be identical to all the other scripts...

```python
# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./data/test.txt')
ap.add_argument('-batch_size', type=int, default=50)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=500)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']
```
