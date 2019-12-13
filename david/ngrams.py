from typing import Iterable, List, Optional, Tuple

import gensim
import sklearn
from texttable import Texttable

from .text.prep import gensim_preprocess, spacy_token_lemmatizer


def sents_to_ngramTokens(
        sentences: List[str],
        min_count: int = 5,
        threshold: int = 100,
        stop_words: Optional[Iterable[str]] = None,
        spacy_pos_tags: Optional[Iterable[str]] = None,
) -> Iterable[List[str]]:
    """Convert texts to n-grams tokens with spaCy and Gensim.

    Removes stopwords, forms (bigrams & trigrams), and lemmatize tokens.
    This method uses the set of STOP_WORDS from spaCy's lang module and the
    gensim.utils.simple_preprocess method, which converts a document
    into a list of lowercase tokens, ignoring tokens that are too short
    or too long.

    Parameters:
    ----------

    `sentences` (List[str]):
        An iterable document of text sequences.

    `min_count` (int or float, default=5):
        Ignore all words and bigrams with total collected count lower than
        this value. Method from class gensim.models.phrases.Phrases.

    `threshold` (float, default=10.0):
        Represent a score threshold for forming the phrases (higher means
        fewer phrases). A phrase of words a followed by b is accepted if
        the score of the phrase is greater than threshold. Heavily depends
        on concrete scoring-function, see the scoring parameter.

    `stop_words` (Optional[Iterable[str]]):
        If None, Spacy's stop-word set will be used.

    `spacy_pos_tags` (Optional[Iterable[str]]):
        Spacy's part-of-speech lemmatizer tags. If None, the default tags will
        be used: ['NOUN', 'ADJ', 'VERB', 'ADV'].

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
        doc: List[str],
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: int = 2000,
        top_num: int = True,
        reverse: bool = True) -> Iterable[Tuple[str, int]]:
    """Extracts ngram frequencies using sklearn's CountVectorizer class.

    Parameters:
    ----------

    `doc` (List[str]):
        An iterable document of text sequences.

    `ngram_range` (tuple, default=(1, 1)):
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used. Source `sklearn.feature_extraction.text.CountVectorizer`

    `max_features` (int or None, default=None):
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.
        This parameter is ignored if vocabulary is not None. Source
        `sklearn.feature_extraction.text.CountVectorizer`

    `top_num` (int or None, default=None):
        If None, it returns an Iterable[List[Tuple[str, int]]] of ngrams and
        its frequencies extracted from the document's vocabulary. Otherwise,
        set a limit value to limit the number of ngrams to return.

    """
    vec = sklearn.feature_extraction.text.CountVectorizer(
        ngram_range=ngram_range, max_features=max_features).fit(doc)
    bow = vec.transform(doc)
    words = bow.sum(axis=0)
    grams = [(w, words[0, i]) for w, i in vec.vocabulary_.items()]
    grams = sorted(grams, key=lambda freq: freq[1], reverse=reverse)
    if isinstance(top_num, int) and top_num > 0:
        return grams[:top_num]
    return grams


def print_ngrams(
        doc: List[str],
        ngrams: Tuple[int, int] = (1, 1),
        top_num: int = 5,
        features: int = 2000) -> None:
    """Prints ngrams and frequencies in a table format."""

    def assign_label(ngram_range: Tuple[int, int]) -> List[str]:
        choices = {1: "uni", 2: "bi", 3: "tri", 4: "quad"}
        xG, yG = ngram_range
        if (xG and yG) not in choices.keys():
            return ["ngrams", "count"]
        else:
            name = ":".join([choices[xG], choices[yG]])
        return [name + "-gram", "count"]

    table = Texttable()
    table.set_cols_align(["l", "r"]).set_cols_valign(["t", "m"])
    grams = n_grams(doc, ngrams, features, top_num, True)
    grams.insert(0, assign_label(ngrams))
    table.add_rows(grams)
    print(table.draw())


def top_unigrams(doc, n=5, ngram_range=(1, 1), max_features=None) -> None:
    print_ngrams(doc, ngram_range, n, max_features)


def top_bigrams(doc, n=5, ngram_range=(2, 2), max_features=2000) -> None:
    print_ngrams(doc, ngram_range, n, max_features)


def top_trigrams(doc, n=5, ngram_range=(3, 3), max_features=2000) -> None:
    print_ngrams(doc, ngram_range, n, max_features)


def top_quadgrams(doc, n=5, ngram_range=(4, 4), max_features=2000) -> None:
    print_ngrams(doc, ngram_range, n, max_features)
