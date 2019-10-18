
import logging
import os

import numpy as np
import pandas as pd
import requests
import spacy
import tensorflow as tf
from IPython.display import HTML
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from spacy import displacy
from spacy.lang.en import English

import tensorflow_hub as hub

logging.getLogger('tensorflow').disabled = True


nlp = spacy.load('en_core_web_lg')


def get_sentences(df: object, position: int = 494):
    '''Preps texts by normalizing spacing and then with spacy
    from a pandas dataframe. The columm name most be named `text`.
    '''
    text = df.iloc[position].text
    text = text.lower().replace('\n', ' ').replace('\t', ' ')
    text = text.replace('\xa0', ' ')
    text = ' '.join(text.split())
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        if len(sent) > 1:
            sentences.append(sent.string.strip())
    return sentences


def semantic_search_engine(search_string: str,
                           sentences: list,
                           num_results: int = 3,
                           embeddings_1: object = None,
                           module_url: str = 'https://tfhub.dev/google/elmo/2'
                           ) -> None:
    '''Sementic Search Engine.
    Enter a set of words to find matching sentences. 'results_returned' can
    be used to modify the number of matching sentences retured. To view the
    code behind this cell, use the menu in the top right to unhide.
    '''
    embed = hub.Module(module_url)
    embeddings_2 = embed(
        [search_string],
        signature='default',
        as_dict=True
    )['default']

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        search_vect = sess.run(embeddings_2)
    CosineSimilarities = pd.Series(
        cosine_similarity(search_vect, embeddings_1).flatten())

    output = ''
    for (i, _) in CosineSimilarities.nlargest(num_results).iteritems():
        output += ('<p style="font-family:verdana; font-size:110%;"> ')
        for i in sentences[i].split():
            if i.lower() in search_string:
                output += (" <b>" + str(i) + "</b>")
            else:
                output += (' ' + str(i))
        output += ("</p><hr>")
    output = ('<h3>Results:</h3>' + output)
    display(HTML(output))


# def create_batch(self, sentence_list):
#     """Create a batch for a list of sentences
#     """
#     embeddings_batch = []

#     for sent in sentence_list:
#         embeddings = []
#         sent_tokens = sent_tokenize(sent)
#         word_tokens = [word_tokenize(w) for w in sent_tokens]
#         tokens = [value for sublist in word_tokens for value in sublist]
#         tokens = [token for token in tokens if token != '']

#         for token in tokens:
#             embeddings.append(self.embdict.tok2emb.get(token))

#         if len(tokens) < self.max_sequence_length:
#             pads = [np.zeros(self.embedding_dim) for _ in range(
#                 self.max_sequence_length - len(tokens)
#             )]
#             embeddings = pads + embeddings
#         else:
#             embeddings = embeddings[-self.max_sequence_length:]

#         embeddings = np.asarray(embeddings)
#         embeddings_batch.append(embeddings)

#     embeddings_batch = np.asarray(embeddings_batch)
#     return embeddings_batch
