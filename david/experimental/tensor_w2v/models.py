import csv
import os
from os.path import join

import gensim
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from ...text.nltk_textpipe import preprocess_doc


class CsvConnector(object):
    def __init__(self,
                 filepath=None,
                 separator=',',
                 text_columns=(),
                 concate_by='. ',
                 preprocessing=None):
        """CSV data loader class."""
        if not text_columns:
            raise ValueError(
                "You have to select at least one column on your input data")

        with open(filepath, 'r', encoding='utf-8') as f:
            self.reader = csv.DictReader(f, delimiter=separator, quotechar='"')
            column_names = self.reader.fieldnames

            for column in text_columns:
                if column not in column_names:
                    print("{} is not a valid column. Found {}".format(
                        column, column_names))

        if not preprocessing:
            def preprocessing(x): return x

        self.filepath = filepath
        self.separator = separator
        self.text_columns = text_columns
        self.concate_by = concate_by
        self.preprocessing = preprocessing

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=self.separator, quotechar='"')

            for line in reader:
                sentence = self.concate_by.join(
                    [line[col] for col in self.text_columns if line[col]])
                yield self.preprocessing(sentence).split()


class TxtConnector(object):
    def __init__(self, filepath=None):
        self.filepath = filepath

    def __iter__(self):
        for text in open(self.filepath, 'r', encoding='utf-8'):
            yield preprocess_doc(text)


class Bigram(object):
    def __init__(self, sentences):
        self.sentences = sentences

    def __iter__(self):
        bigram = gensim.models.Phrases(self.sentences)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        for sent in self.sentences:
            yield bigram_mod[sent]


class Word2Vec(object):
    def __init__(self, model=None, save_folder=None, phrases=False):
        self.model = model
        self.save_folder = join(save_folder, 'gensim-model.cpkt')
        self.phrases = phrases

    def fit(self, *args, **kwargs):
        self.model = gensim.models.Word2Vec(*args, **kwargs)
        self.model.save(self.save_folder)


def create_embeddings(gensim_model, model_folder, trainable=False):
    weights = gensim_model.wv.vectors
    idx2words = gensim_model.wv.index2word
    vocab_size = weights.shape[0]
    embedding_dim = weights.shape[1]

    with open(join(model_folder, 'metadata.tsv'), 'w') as tsv_file:
        tsv_file.writelines('\n'.join(idx2words))

    tf.compat.v1.reset_default_graph()

    X = tf.constant(0.0, shape=[vocab_size, embedding_dim])
    W = tf.Variable(X, trainable=trainable, name='W')
    embedding_placeholder = tf.compat.v1.placeholder(
        tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    writer = tf.compat.v1.summary.FileWriter(
        model_folder, graph=tf.compat.v1.get_default_graph())
    saver = tf.compat.v1.train.Saver()
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = join(model_folder, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    with tf.compat.v1.Session() as sess:
        tf_model_filepath = join(model_folder, 'tf-model.cpkt')
        sess.run(embedding_init, feed_dict={embedding_placeholder: weights})
        save_path = saver.save(sess, tf_model_filepath)
    return save_path


class Embeddings(Bigram, Word2Vec):
    def __init__(gensim_model, model_folder):
        self.gensim_model = gensim_model
        self.model_folder = model_folder
        pass
