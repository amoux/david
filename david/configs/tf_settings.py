import os
from multiprocessing import cpu_count

from ._parser import read_ini_config

TENSORBOARD_DEFAULT_SETTINGS_INI = '''
[Corpus]
corpus_path: {corpus_path}

[Model]
model_name: {model_name}
model_path: {model_path}

[Word2Vec]
sentences: None
size: 100
alpha: 0.025
window: 5
min_count: 5
max_vocab_size: None
sample: 1e-3
seed: 1
workers:  {workers}
min_alpha: 0.0001
sg: 0
hs: 0
negative: 10
cbow_mean: 1
iter: iter
null_word: 0'''


def cfg_tensorboard_ini(corpus_path, model_name=None,
                        workers=None, root_dirname='models'):
    if not workers:
        workers = cpu_count()

    corpus_path = os.path.realpath(corpus_path)
    abs_model_path = os.path.join(
        os.path.realpath(f'{root_dirname}/' + model_name))
    os.makedirs(abs_model_path, exist_ok=True)

    tb_ini_cfgs = TENSORBOARD_DEFAULT_SETTINGS_INI.format(
        corpus_path=corpus_path,
        model_name=model_name,
        model_path=abs_model_path,
        workers=workers)

    ini_file_path = os.path.join(abs_model_path, f'{model_name}_model_.ini')
    return read_ini_config(ini_file_path, default_ini_cfg=tb_ini_cfgs)
