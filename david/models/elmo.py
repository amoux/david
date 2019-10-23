import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

ELMO2_MODULE_URL = 'https://tfhub.dev/google/elmo/2'


def model_file(model: str = None, dirpath='models', override=False):
    """Model file manager for .npy files, are platform-independent
    and are compact and faster to load and save models.

    Parameters:
    ----------
    model : (str)
        A string from the name of the var object
        is used for saving and loading the npy file.

    Usage:
    -----
        ...
        >>> elmo_embeddings = sess.run(embeddings)
        >>> model_file('elmo_embeddings')
    elmo_embeddings obj saved @: models/elmo_embeddings.npy

    Returns a numpy.ndarray if the model name and path exist.
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    file_path = os.path.join(dirpath, f'{model}.npy')
    file_exist = os.path.exists(file_path)
    model_obj = eval(f'{model}')
    if not file_exist:
        print(f'{model} obj saved @: {file_path}')
        np.save(file_path, model_obj)
    elif file_exist and override:
        print(f'{model} obj overridden @: {file_path}')
        np.save(file_path, model_obj)
    else:
        return np.load(file_path)


def embed_module(iterable_obj: list):
    embed = hub.Module(ELMO2_MODULE_URL)
    return embed(iterable_obj, signature='default', as_dict=True)['default']


def embed_sentences(sents: list):
    """Create elmo embeddings from sentences."""
    embed_sents = embed_module(sents)

    session_conf = tf.compat.v1.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False)

    with tf.compat.v1.Session(config=session_conf) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        embed_sents = sess.run(embed_sents)
    return embed_sents


def elmo_search(query: str, n_results: int, sents: list, embed_sents: object):
    """ELmo Embeddings Sementic Search Engine."""
    embed_query = embed_module([query])

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        embed_query = sess.run(embed_query)

    similar_docs = pd.Series(
        cosine_similarity(embed_query, embed_sents).flatten())

    output = ''
    for i, j in similar_docs.nlargest(n_results).iteritems():
        output += ('<p style="font-family:verdana; font-size:110%;"> ')
        for i in sents[i].split():
            if i.lower() in query:
                output += (" <b>" + str(i) + "</b>")
            else:
                output += (' ' + str(i))
        output += ("</p><hr>")
    output = ('<h3>Results:</h3>' + output)
    display(HTML(output))
