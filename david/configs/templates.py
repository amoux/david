"""TEMPLATES FOR VARIOUS INI-FILE CONFIGURATIONS.

TODOS:
* The following descriptions are the different templates I need
to add here to make it easier to save the various settings used
with experimenting with models.

    - database files
    - gensim w2v models
    - TFIDF models
    - Tensorflow-Hub models (bert, elmo ect.)
    - Tensorboard word embeddings
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
