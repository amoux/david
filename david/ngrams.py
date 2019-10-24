from typing import Iterable, List

import gensim
import spacy
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer

from .lang import SPACY_STOP_WORDS


def gensim_preprocess(stopwords, sentences):
    return [[word for word in simple_preprocess(sent)
             if word not in stopwords] for sent in sentences]


def spacy_preprocess(docs, spacy_model, pos_tags, disable_pipe_names):
    nlp = spacy.load(spacy_model, disable=disable_pipe_names)
    prep_docs = []
    for doc in docs:
        doc = nlp(' '.join(doc))
        prep_docs.append(
            [token.lemma_ for token in doc if token.pos_ in pos_tags])
    return prep_docs


def text2ngrams(sentences: Iterable[str],
                spacy_model: str = None,
                pos_tags: list = None,
                stop_words: set = None,
                disable_pipe_names: list = None,
                min_count: int = 5,
                threshold: float = 10.0) -> List[str]:
    """Convert texts to N-grams (Spacy & Gensim).

    Removes stopwords, forms (bigrams & trigrams), and lemmatizes texts.
    This method uses the set of STOP_WORDS from spaCy's lang module and the
    gensim.utils.simple_preprocess method, which converts a document
    into a list of lowercase tokens, ignoring tokens that are too short
    or too long.

    Default configurations if left as none:

        * `spacy_model = 'en_core_web_lg'`
        * `disable_pipe_names = ['parser', 'ner']`
        * `stop_words = SPACY_STOP_WORDS`
        * `pos_tags = ['NOUN', 'ADJ', 'VERB', 'ADV']`

    Parameters:
    ----------
    sentences : (iterable of str)
        A processed text data with: data_ready = text2ngrams(sentences).
    spacy_model : (spacy language model, default='en_core_web_lg')
        Other spacy models are compatible with this function.
    min_count : (Type[int|float], default=5)
        Ignore all words and bigrams with total collected count lower than
        this value. Method from class gensim.models.phrases.Phrases.
    threshold : (float, default=10.0)
         Represent a score threshold for forming the phrases (higher means
         fewer phrases). A phrase of words a followed by b is accepted if
         the score of the phrase is greater than threshold. Heavily depends
         on concrete scoring-function, see the scoring parameter.

    Returns:
    -------
        Returns preprocessed texts.
    """
    if not spacy_model:
        spacy_model = 'en_core_web_lg'
    if not disable_pipe_names:
        disable_pipe_names = ['parser', 'ner']
    if not pos_tags:
        pos_tags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    if not stop_words:
        stop_words = SPACY_STOP_WORDS
    # build biagrams and trigram models
    bigram = gensim.models.Phrases(sentences, min_count, threshold)
    trigram = gensim.models.Phrases(bigram[sentences], threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    texts = gensim_preprocess(stop_words, sentences)
    texts = [bigram_mod[text] for text in texts]
    texts = [trigram_mod[bigram_mod[text]] for text in texts]
    texts = spacy_preprocess(texts, spacy_model, pos_tags,
                             disable_pipe_names)
    # remove stopwords once more after lemmatization
    texts = gensim_preprocess(stop_words, texts)
    return texts


def n_grams(corpus: list,
            n: int = 5,
            ngram_range: tuple = None,
            max_features: int = None,
            reverse: bool = True):
    """N-gram Frequency with Sklearn CountVectorizer.

    Returns the most frequently occurring words.
    """
    if not ngram_range:
        ngram_range = (1, 1)

    vec = CountVectorizer(ngram_range=ngram_range,
                          max_features=max_features).fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]

    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=reverse)
    return wordfreq[:n]


def top_unigrams(corpus, n=5, ngram_range=(1, 1),
                 max_features=None, reverse=True):
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_bigrams(corpus, n=5, ngram_range=(2, 2),
                max_features=2000, reverse=True):
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_trigrams(corpus, n=5, ngram_range=(3, 3),
                 max_features=2000, reverse=True):
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_quadgrams(corpus, n=5, ngram_range=(4, 4),
                  max_features=2000, reverse=True):
    return n_grams(corpus, n, ngram_range, max_features, reverse)
