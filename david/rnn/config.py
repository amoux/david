from types import SimpleNamespace as _SimpleNamespace


# TEST_FILE = 'test.txt'
TEST_FILE = 'older/shakespeare_input.txt'

_defaults = {
    'data_dir': TEST_FILE,
    'batch_size': 50,
    'layer_num': 2,
    'seq_length': 50,
    'hidden_dim': 500,
    'generate_length': 500,
    'nb_epoch': 20,
    'mode': 'train',
    'weights': '',
}


def network():
    """(NET)Returns the parameters for the network definition.
    """
    return _SimpleNamespace(**_defaults)
