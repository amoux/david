from sklearn.feature_extraction.text import CountVectorizer


def n_grams(corpus: list,
            n: int = 5,
            ngram_range: tuple = (1, 1),
            max_features: int = None,
            reverse: bool = True):
    '''
    N-Gram CountVectorizer for the most frequently occurring words.

        * unigrams-range: (1, 1)
        * biagrams-range: (2, 2)
        * trigrams-range: (3, 3)
        * quadgram-range: (4, 4)
    '''

    if isinstance(ngram_range, tuple):
        vec = CountVectorizer(ngram_range=ngram_range,
                              max_features=max_features
                              ).fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=reverse)
    return wordfreq[:n]


def top_unigrams(corpus, n=5, max_features=None, reverse=True):
    '''N-Gram CountVectorizer most frequently used Unigrams.

        >>> top_words = top_unigrams(corpus, n=5)
        >>> top_df = pd.DataFrame(top_words)
        >>> top_df.columns = ["unigram", "frequency"]
    '''
    return n_grams(corpus, n, max_features, reverse)


def top_bigrams(corpus, n=5, ngram_range=(2, 2),
                max_features=2000, reverse=True):
    '''N-Gram CountVectorizer most frequently used Bi-grams.

        >>> bigram_words = top_bigrams(corpus, n=5)
        >>> bigram_df = pd.DataFrame(top_bigram_words)
        >>> bigram_df.columns = ["bi-gram", "frequency"]
        >>> print(top_bigram_df)
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_trigrams(corpus, n=5, ngram_range=(3, 3),
                 max_features=2000, reverse=True):
    '''N-Gram CountVectorizer most frequently used Tri-grams.

        >>> triagram_words = top_trigrams(corpus, n=5)
        >>> triagram_df = pd.DataFrame(top_triagram_words)
        >>> triagram_df.columns = ["tri-gram", "frequency"]
        >>> print(top_triagram_df)
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)


def top_quadgrams(corpus, n=5, ngram_range=(4, 4),
                  max_features=2000, reverse=True):
    '''N-Gram CountVectorizer most frequently used Quad-grams.

    >>> quadgram_words = top_quadgrams(corpus, n=5)
    >>> quadgram_df = pd.DataFrame(top_quadgram_words)
    >>> quadgram_df.columns = ["quad-gram", "frequency"]
    >>> print(top_quadgram_df)
    '''
    return n_grams(corpus, n, ngram_range, max_features, reverse)
