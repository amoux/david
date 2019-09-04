from typing import Iterable, List

import gensim
import spacy
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from spacy import lang

SPACY_MODEL = 'en_core_web_lg'
SPACY_DISABLE = ['parser', 'ner']
SPACY_POSTAGS = ['NOUN', 'ADJ', 'VERB', 'ADV']
STOP_WORDS = lang.en.stop_words.STOP_WORDS


def _prep(stopwords: set, sentences: list):
    # removes stop-words and preprocess texts with gensim.
    texts = [[word for word in simple_preprocess(str(sent))
              if word not in stopwords] for sent in sentences]
    return texts


def text2ngrams(sentences: Iterable[str],
                spacy_model: str = SPACY_MODEL,
                disable: list = SPACY_DISABLE,
                stop_words: set = STOP_WORDS,
                post_tags: list = SPACY_POSTAGS,
                min_count: int = 5,
                threshold: float = 10.0
                ) -> List[str]:
    '''
    Convert texts to N-grams (Spacy & Gensim).

    Removes stopwords, forms (bigrams & trigrams), and lemmatizes texts.
    This method uses the set of STOP_WORDS from spaCy's lang module and the
    gensim.utils.simple_preprocess method, which converts a document
    into a list of lowercase tokens, ignoring tokens that are too short
    or too long.

    NOTE: The following default parameters can be overwritten:

        * SPACY_MODEL = 'en_core_web_lg'
        * SPACY_DISABLE = ['parser', 'ner']
        * STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS
        * SPACY_POSTAGS = ['NOUN', 'ADJ', 'VERB', 'ADV']

    Parameters:
    ----------

    `sentences` : (iterable of str)
        A processed text data with: data_ready = text2ngrams(sentences).

    `spacy_model` : (spacy language model, default='en_core_web_lg')
        Other spacy models are compatible with this function.

    `min_count` (Type[int|float], default=5)
        Ignore all words and bigrams with total collected count lower than
        this value. Method from class `gensim.models.phrases.Phrases`

    `threshold` (float, default=10.0)
         Represent a score threshold for forming the phrases (higher means
         fewer phrases). A phrase of words a followed by b is accepted if
         the score of the phrase is greater than threshold. Heavily depends
         on concrete scoring-function, see the scoring parameter. Method from
         class `gensim.models.phrases.Phrases`

    Returns:
    -------
        Returns preprocessed texts.

    '''
    if spacy_model:
        SPACY_MODEL = spacy_model
    if disable:
        SPACY_DISABLE = disable
    if stop_words:
        STOP_WORDS = stop_words
    if post_tags:
        SPACY_POSTAGS = post_tags

    # build biagrams and trigram models
    bigram = gensim.models.Phrases(sentences, min_count, threshold)
    trigram = gensim.models.Phrases(bigram[sentences], threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = _prep(STOP_WORDS, sentences)
    texts = [bigram_mod[text] for text in texts]
    texts = [trigram_mod[bigram_mod[text]] for text in texts]

    nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLE)
    spacy.prefer_gpu()

    texts_out = []
    for sent in texts:
        doc = nlp(' '.join(sent))
        texts_out.append(
            [
                token.lemma_ for token in doc
                if token.pos_ in SPACY_POSTAGS
            ]
        )
    # remove stopwords once more after lemmatization
    texts_out = _prep(STOP_WORDS, texts_out)
    return texts_out


def n_grams(corpus: list,
            n: int = 5,
            ngram_range: tuple = (1, 1),
            max_features: int = None,
            reverse: bool = True):
    '''
    N-gram Frequency with Sklearn CountVectorizer.

    Returns the most frequently occurring words.

        * unigrams-range: (1, 1)
        * biagrams-range: (2, 2)
        * trigrams-range: (3, 3)
        * quadgram-range: (4, 4)
    '''

    if not isinstance(ngram_range, tuple):
        raise ValueError(f'{ngram_range} is not a tuple type = tuple(n, n)')

    vec = CountVectorizer(ngram_range=ngram_range,
                          max_features=max_features
                          ).fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=reverse)
    return wordfreq[:n]


def top_unigrams(corpus, n=5, ngram_range=(1, 1),
                 max_features=None, reverse=True):
    '''N-Gram CountVectorizer most frequently used Unigrams.

    Usage:
    -----

        >>> top_words = top_unigrams(corpus, n=5)
        >>> top_df = pd.DataFrame(top_words)
        >>> top_df.columns = ["unigram", "frequency"]
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_bigrams(corpus, n=5, ngram_range=(2, 2),
                max_features=2000, reverse=True):
    '''N-Gram CountVectorizer most frequently used Bi-grams.

    Usage:
    -----

        >>> bigram_words = top_bigrams(corpus, n=5)
        >>> bigram_df = pd.DataFrame(top_bigram_words)
        >>> bigram_df.columns = ["bi-gram", "frequency"]
        >>> print(top_bigram_df)
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_trigrams(corpus, n=5, ngram_range=(3, 3),
                 max_features=2000, reverse=True):
    '''N-Gram CountVectorizer most frequently used Tri-grams.

    Usage:
    -----

        >>> triagram_words = top_trigrams(corpus, n=5)
        >>> triagram_df = pd.DataFrame(top_triagram_words)
        >>> triagram_df.columns = ["tri-gram", "frequency"]
        >>> print(top_triagram_df)
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_quadgrams(corpus, n=5, ngram_range=(4, 4),
                  max_features=2000, reverse=True):
    '''N-Gram CountVectorizer most frequently used Quad-grams.

    Usage:
    -----

        >>> quadgram_words = top_quadgrams(corpus, n=5)
        >>> quadgram_df = pd.DataFrame(top_quadgram_words)
        >>> quadgram_df.columns = ["quad-gram", "frequency"]
        >>> print(top_quadgram_df)
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)
