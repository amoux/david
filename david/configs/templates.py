"""
Configuration file templates.

-----------------------------
The templates in this module are used mostly by models
where many pararameters are needed to initialize.
"""

INI_TEMPLATE_TENSORBOARD = """
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
null_word: 0"""

INI_TEMPLATE_YTSENTIMENT = """
[Corpus]
max_strlen: {max_strlen}
min_strlen: {min_strlen}
spacy_model: {spacy_model}
[Tokenizer]
max_seqlen: {max_seqlen}
glove_ndim: {glove_ndim}
vocab_shape: {vocab_shape}
[Model]
activation: {activation}
trainable: {trainable}
epochs: {epochs}
loss: {loss}
optimizer: {optimizer}
padding: {padding}
[Output]
project_dir: {project_dir}
model_file: {model_file}
vocab_file: {vocab_file}
config_file: {config_file}"""
