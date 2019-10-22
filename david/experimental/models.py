import csv
import os

import gensim
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def join_paths(dirname, filename):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return os.path.join(dirname, filename)


class CsvConnector(object):
    def __init__(
            self,
            filepath=None,
            separator=',',
            text_columns=(),
            columns_joining_token='. ',
            preprocessing=None,
    ):
        """CSV data loader class."""
        if not text_columns:
            print("You have to select at least one column on your input data")
            raise

        with open(filepath, 'r', encoding='utf-8') as f:
            self.reader = csv.DictReader(f, delimiter=separator, quotechar='"')
            column_names = self.reader.fieldnames
            for column in text_columns:
                if column not in column_names:
                    print("{} is not a valid column. Found {}".format(
                        column, column_names))
                    raise

        if not preprocessing:
            def preprocessing(x): return x

        self.filepath = filepath
        self.separator = separator
        self.text_columns = text_columns
        self.columns_joining_token = columns_joining_token
        self.preprocessing = preprocessing

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=self.separator, quotechar='"')
            for line in reader:
                sentence = self.columns_joining_token.join(
                    [line[col] for col in self.text_columns if line[col]])
                yield self.preprocessing(sentence).split()


class TxtConnector(object):
    def __init__(self, filepath=None, preprocessing=None):
        if not preprocessing:
            def preprocessing(x): return x
        self.filepath = filepath
        self.preprocessing = preprocessing

    def __iter__(self):
        for line in open(self.filepath, 'r', encoding='utf-8'):
            yield self.preprocessing(line).split()


class Bigram(object):
    def __init__(self, iterator):
        self.iterator = iterator
        self.bigram = gensim.models.Phrases(self.iterator)

    def __iter__(self):
        for sentence in self.iterator:
            yield self.bigram[sentence]


class Word2Vec(object):
    def __init__(self, model=None, save_folder=None, phrases=False):
        self.model = model
        self.save_folder = join_paths(save_folder, 'gensim-model.cpkt')
        self.phrases = phrases

    def fit(self, *args, **kwargs):
        self.model = gensim.models.Word2Vec(*args, **kwargs)
        self.model.save(self.save_folder)


def create_embeddings(gensim_model, model_folder, trainable=False):
    weights = gensim_model.wv.vectors
    idx2words = gensim_model.wv.index2word

    vocab_size = weights.shape[0]
    embedding_dim = weights.shape[1]

    metadata_filepath = join_paths(model_folder, 'metadata.tsv')

    with open(metadata_filepath, 'w') as tsv_file:
        tsv_file.writelines('\n'.join(idx2words))

    tf.reset_default_graph()

    X = tf.constant(0.0, shape=[vocab_size, embedding_dim])
    W = tf.Variable(X, trainable=trainable, name='W')
    embedding_placeholder = tf.placeholder(
        tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)

    writer = tf.summary.FileWriter(model_folder, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    config = projector.ProjectorConfig()

    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = metadata_filepath
    projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        tf_model_filepath = join_paths(model_folder, 'tf-model.cpkt')
        sess.run(embedding_init, feed_dict={embedding_placeholder: weights})
        save_path = saver.save(sess, tf_model_filepath)
    return save_path


class Embeddings(Bigram, Word2Vec):
    def __init__(gensim_model, model_folder):
        self.gensim_model = gensim_model
        self.model_folder = model_folder
        pass
