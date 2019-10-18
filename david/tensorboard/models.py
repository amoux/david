import csv
import os
import parser

import gensim
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from .train import preprocessing


class CsvConnector(object):
    CSV_READER = None
    FILE_PATH = None

    def __init__(
        self,
        filepath=None,
        separator=',',
        quotechar='"',
        df_cols: tuple = ('text',),
        col_join_token='. '
    ):

        self.file_path = filepath
        self.df_cols = df_cols
        self.separator = separator
        self.quotechar = quotechar
        self.col_join_token = col_join_token

    def load_csv(self, fp, mode='r'):
        with open(fp, mode, encoding='uft-8') as f:
            self.CSV_READER = csv.DictReader(f, self.separator, self.quotechar)

            self.FILE_PATH = fp
            loded_cols = self.CSV_READER.fieldnames

            for col in self.df_cols:
                if col in loded_cols:
                    print(f'Yay selecting column < {col} >.')
                else:
                    raise ValueError('Ouch you need to select: (text or csv)')

    def __iter__(self):
        """CSV iterator loads the file_path from the
        path passed to the parameter. At the time of
        loading the path is alson passed to the FILE_PATH
        class parameter.
        """
        self.load_csv(fp=self.file_path)

        for line in self.CSV_READER:
            sentence = self.col_join_token.join(
                [line[col] for col in self.df_cols if line[col]]
            )
            yield preprocessing(sentence).split()


class TxtConnector(object):
    def __init__(self, file_path=None):
        self.file_path = file_path

    def __iter__(self):
        for line in open(self.file_path, 'r', encoding='utf-8'):
            yield preprocessing(line).split()


class Bigram(object):
    """Gensim Biagram Class."""

    def __init__(self, sentences):
        self.sentences = sentences
        self.bigram = gensim.models.Phrases(self.sentences)

    def __iter__(self):
        for sent in self.sentences:
            yield self.bigram[sent]


class Word2Vec(object):
    def __init__(self, model=None, save_folder=None, phrases=False):
        """Word2Vec Class Model.

        Parameters:
        ----------
        model : (object, default=None)
        save_folder : (str)
            The directory name where the model's output will be saved.
        """
        if not os.path.exists(save_folder):
            print(f"{save_folder}: Folder does not exist, create it first")

        self.model = model
        self.save_folder = save_folder
        self.phrases = phrases

    def fit(self, sentences, size=100, alpha=0.025,
            window=5, min_count=5, max_vocab_size=None,
            sample=1e-3, seed=1, workers=4, min_alpha=0.0001,
            sg=0, hs=0, negative=10, cbow_mean=1, iter=5, null_word=0):
        self.model = gensim.models.Word2Vec(
            sentences,
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
        self.model.save(os.path.join(self.save_folder, "gensim-model.cpkt"))


def create_embeddings(gensim_model=None, MODEL_FOLDER=None):
    weights = gensim_model.wv.vectors
    idx2words = gensim_model.wv.index2word
    vocab_size = weights.shape[0]
    embedding_dim = weights.shape[1]

    with open(os.path.join(MODEL_FOLDER, "metadata.tsv"), 'w') as f:
        f.writelines("\n".join(idx2words))
    tf.reset_default_graph()

    W = tf.Variable(tf.constant(
        0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
    embedding_placeholder = tf.placeholder(
        tf.float32, [vocab_size, embedding_dim])

    embedding_init = W.assign(embedding_placeholder)
    writer = tf.summary.FileWriter(MODEL_FOLDER, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    # Format for projector.ProjectorConfig():
    # tensorflow/contrib/tensorboard/plugins/projector/projector_config.proto

    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.

    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = os.path.join(MODEL_FOLDER, "metadata.tsv")

    # Saves a configuration file that TensorBoard will read during startup.

    projector.visualize_embeddings(writer, config)

    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: weights})
        save_path = saver.save(sess, os.path.join(
            MODEL_FOLDER, "tf-model.cpkt"))
    return save_path

    parser.add_argument("--folder", default="models/movie_reviews")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--sample", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min_alpha", type=float, default=0.0001)
    parser.add_argument("--sg", type=int, default=0)
    parser.add_argument("--hs", type=int, default=0)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--cbow_mean", type=int, default=1)
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--null_word", type=int, default=0)


if __name__ == "__main__":
    import json
    import shutil
    from types import SimpleNamespace
    params = SimpleNamespace(folder="testw2v", size=20, alpha=0.025,
                             window=5, min_count=5, max_vocab_size=None,
                             sample=1e-3, seed=1, workers=3,
                             min_alpha=0.0001, sg=0, hs=0,
                             negative=10, cbow_mean=1, iter=5, null_word=0)
    unigram_generator = TxtConnector(filepath="data/SMSSpamCollection.txt")
    sentence_generator = Bigram(unigram_generator)
    os.makedirs(params.folder)
    json.dump(vars(params), open(
        os.path.join(
            params.folder, "params.json"
        ), 'w', encoding='utf-8'), indent=2)
    # create a Word2Vec instance.
    w2v = Word2Vec(save_folder=params.folder)
    # pass the parameters to the Word2Vec method
    w2v.fit(
        sentence_generator,
        size=params.size,
        alpha=params.alpha,
        window=params.window,
        min_count=params.min_count,
        max_vocab_size=params.max_vocab_size,
        sample=params.sample,
        seed=params.seed,
        workers=params.workers,
        min_alpha=params.min_alpha,
        sg=params.sg,
        hs=params.hs,
        negative=params.negative,
        cbow_mean=params.cbow_mean,
        iter=params.iter,
        null_word=params.null_word)
    create_embeddings(gensim_model=w2v.model, MODEL_FOLDER=params.folder)
    shutil.rmtree(params.folder)
