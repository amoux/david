import csv
import os

import gensim
import tensorflow as tf
from tensorboard.plugins import projector
from tensorflow.contrib import tensorboard


class CsvConnector(object):
    def __init__(self,
                 fpath: str = None,
                 delimiter: str = ',',
                 text_col: tuple = (),
                 join_col_by: str = '. ',
                 preprocessing: bool = None
                 ):
        if not text_col:
            raise ValueError(
                'You have to select at least one column on your input data.')

        with open(fpath, 'r', encoding='utf-8') as csv_file:
            self.reader = csv.DictReader(csv_file, delimiter=delimiter,
                                         quotechar='"')
            columns = self.reader.fieldnames
            for col in text_col:
                if col not in columns:
                    raise ValueError(
                        f'{col} is not a valid column. Found {columns}')

        self.fpath = fpath
        self.delimiter = delimiter
        self.text_col = text_col
        self.join_col_by = join_col_by
        self.preprocessing = preprocessing

    def __iter__(self):
        with open(self.fpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=self.delimiter, quotechar='"')
            for line in reader:
                sentence = self.join_col_by.join(
                    [line[col] for col in self.text_col if line[col]])
                yield self.preprocessing(sentence).split()


class TxtConnector(object):
    def __init__(self, fpath=None, preprocessing=None):
        self.fpath = fpath
        self.preprocessing = preprocessing

    def __iter__(self):
        for line in open(self.fpath, 'r', encoding='utf-8'):
            yield self.preprocessing(line).split()


class Bigram(object):

    def __init__(self, sentences, delimiter=b'_'):
        self.sentences = sentences
        self.delimiter = delimiter
        self.bigram = gensim.models.Phrases(
            self.sentences, delimiter=self.delimiter)

    def __iter__(self):
        for sent in self.sentences:
            yield self.bigram[sent]


class Word2Vec(object):
    def __init__(self,
                 sentences=None,
                 model=None,
                 save_folder=None,
                 model_name='gensim-model.cpkt'
                 ):
        '''Word2Vec Class Model.
        '''
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        self.sentences = sentences
        self.model = model
        self.save_folder = save_folder
        self.model_name = model_name

    def fit(self, size=100, alpha=0.025,
            window=5, min_count=5, max_vocab_size=None,
            sample=1e-3, seed=1, workers=4, min_alpha=0.0001,
            sg=0, hs=0, negative=10, cbow_mean=1, iter=5, null_word=0
            ):
        self.model = gensim.models.Word2Vec(self.sentences,
                                            size=size,
                                            alpha=alpha,
                                            window=window,
                                            min_count=min_count,
                                            max_vocab_size=max_vocab_size,
                                            sample=sample,
                                            seed=seed,
                                            workers=workers,
                                            min_alpha=min_alpha,
                                            sg=sg,
                                            hs=hs,
                                            negative=negative,
                                            cbow_mean=cbow_mean,
                                            iter=iter,
                                            null_word=null_word)
        self.model.save(os.path.join(self.save_folder, self.model_name))


def check_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def write2file(index2word, meta_name: str, save_path: str, join_by='\n'):
    check_dir(save_path)
    with open(os.path.join(save_path, meta_name), 'w') as f:
        f.writelines(join_by.join(index2word))


def save_embedding_config(tensor_name, tf_writer,
                          meta_name: str, save_path: str):
    '''
    TensorBoard embeddings visualizer configuration session.
    Adds format for the projector embeddings and saves the
    configuration file that TensorBoard will read during startup.

    Path:
    `tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto`
    '''
    config = tensorboard.plugins.projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tensor_name
    embedding.metadata_path = os.path.join(save_path, meta_name)
    tensorboard.plugins.projector.visualize_embeddings(tf_writer, config)


def evaluate_tensor_fetches(tensor, placeholder, vectors,
                            tf_saver, model_name, save_path):
    '''
    Runs operations and evaluates tensors in fetches.
    This method runs one "step" of TensorFlow computation, by running the
    necessary graph fragment to execute every Operation and evaluate every
    Tensor in fetches, substituting the values in feed_dict for the
    corresponding input values.

    Returns a model's path from the session's instance.
    '''
    with tf.Session() as sess:
        sess.run(tensor, feed_dict={placeholder: vectors})
        # saves a configuration file that TensorBoard will read during startup.
        return tf_saver.save(sess, os.path.join(save_path, model_name))


def create_embeddings(gensim_model,
                      tf_value=0.0,
                      tf_trainable=False,
                      tf_varname='W',
                      model_name='tf-model.cpkt',
                      meta_name='metadata.tsv',
                      save_path='models/',
                      join_by='\n'):
    '''
    TensorBoard embedding visualizer for a gensim models.

    Parameters:
    ----------

    `tf_value` : (int|list, default=0.0)
        A constant value (or list) of output type `dtype`.

    `tf_varname` : (str, default='W')
        Optional name for the tensor.

    '''
    vectors = gensim_model.wv.vectors
    index2word = gensim_model.wv.index2word
    vocab_size = vectors.shape[0]
    embedding = vectors.shape[1]

    write2file(index2word, meta_name=meta_name,
               save_path=save_path, join_by=join_by)

    tf.reset_default_graph()
    W = tf.Variable(
        tf.constant(tf_value, shape=[vocab_size, embedding]),
        trainable=tf_trainable,
        name=tf_varname
    )
    writer = tf.summary.FileWriter(save_path, graph=tf.get_default_graph())
    save_embedding_config(W.name, writer, meta_name, save_path)
    placeholder = tf.placeholder(tf.float32, [vocab_size, embedding])
    save_path = evaluate_tensor_fetches(tensor=W.assign(placeholder),
                                        placeholder=placeholder,
                                        vectors=vectors,
                                        tf_saver=tf.train.Saver(),
                                        model_name=model_name,
                                        save_path=save_path)
    return save_path
