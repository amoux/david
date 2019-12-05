import gensim
import sklearn

from .text.prep import gensim_preprocess, spacy_token_lemmatizer


def sents_to_ngramTokens(
        sentences: list,
        min_count=5,
        threshold=100,
        stop_words=None,
        spacy_pos_tags=None,
):
    """Convert texts to n-grams tokens with spaCy and Gensim.

    Removes stopwords, forms (bigrams & trigrams), and lemmatizes tokens.
    This method uses the set of STOP_WORDS from spaCy's lang module and the
    gensim.utils.simple_preprocess method, which converts a document
    into a list of lowercase tokens, ignoring tokens that are too short
    or too long.

    Parameters:
    ----------

    `sentences` (Iterable[str]):
        A processed text data with: data_ready = text2ngrams(sentences).

    `min_count` (Type[int|float], default=5):
        Ignore all words and bigrams with total collected count lower than
        this value. Method from class gensim.models.phrases.Phrases.

    `threshold` (float, default=10.0):
        Represent a score threshold for forming the phrases (higher means
        fewer phrases). A phrase of words a followed by b is accepted if
        the score of the phrase is greater than threshold. Heavily depends
        on concrete scoring-function, see the scoring parameter.

    `stop_words` (Type[list|set]):
        If None, spaCy's stop-word set will be used.

    `spacy_pos_tags` (Type[list|set]):
        spaCy's part of speech tags to use for lemmatizing word sequences.
        If None, the default tags will be used: ['NOUN', 'ADJ', 'VERB', 'ADV'].

    Returns a list of preprocessed ngram-tokens.

    """
    # build biagrams and trigram models
    bigram = gensim.models.Phrases(sentences, min_count, threshold)
    trigram = gensim.models.Phrases(bigram[sentences], threshold=threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    tokens = gensim_preprocess(sentences, stop_words)
    tokens = [bigram_mod[tok] for tok in tokens]
    tokens = [trigram_mod[bigram_mod[tok]] for tok in tokens]
    tokens = spacy_token_lemmatizer(tokens, spacy_pos_tags)
    return tokens


def n_grams(
    doc: list,
    n=5,
    ngram_range=None,
    max_features=None,
    reverse=True,
):
    """N-gram Frequency with Sklearn CountVectorizer."""
    if not ngram_range:
        ngram_range = (1, 1)
    vec = sklearn.feature_extraction.text.CountVectorizer(
        ngram_range=ngram_range, max_features=max_features).fit(doc)
    bag_of_words = vec.transform(doc)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(w, wordsums[0, i]) for w, i in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=reverse)
    return wordfreq[:n]


def top_unigrams(doc, n=5, ngram_range=(1, 1),
                 max_features=None, reverse=True):
    return n_grams(doc, n, ngram_range, max_features, reverse)


def top_bigrams(doc, n=5, ngram_range=(2, 2),
                max_features=2000, reverse=True):
    return n_grams(doc, n, ngram_range, max_features, reverse)


def top_trigrams(doc, n=5, ngram_range=(3, 3),
                 max_features=2000, reverse=True):
    return n_grams(doc, n, ngram_range, max_features, reverse)


def top_quadgrams(doc, n=5, ngram_range=(4, 4),
                  max_features=2000, reverse=True):
    return n_grams(doc, n, ngram_range, max_features, reverse)
