from sklearn.feature_extraction.text import CountVectorizer


def get_top_unigrams(corpus, topn=None):
    '''(CountVectorizer) Most Frequently Occurring Unigrams.

    EXAMPLE
    -------
    >>> top_words = get_top_unigrams(corpus, n=20)
    >>> top_df = pd.DataFrame(top_words)
    >>> top_df.columns = ["unigram", "frequency"]
    '''
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:topn]


def get_top_bigrams(corpus, topn=20):
    '''(CountVectorizer) Most Frequently Occurring Bi-grams.

    EXAMPLE
    -------
    Convert the most frequently bigram words to a df
    for example to plot words to a bar plot.

    >>> top_bigram_words = get_top_bigrams(corpus, n=20)
    >>> top_bigram_df = pd.DataFrame(top_bigram_words)
    >>> top_bigram_df.columns = ["bi-gram", "frequency"]
    >>> print(top_bigram_df)
    '''
    vectorizer = CountVectorizer(ngram_range=(
        2, 2), max_features=2000).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:topn]


def get_top_trigrams(corpus, topn=20):
    '''(CountVectorizer) Most Frequently Used Tri-grams.

    EXAMPLE
    -------
    >>> top_triagram_words = get_top_trigrams(corpus, n=20)
    >>> top_triagram_df = pd.DataFrame(top_triagram_words)
    >>> top_triagram_df.columns = ["tri-gram", "frequency"]
    >>> print(top_triagram_df)
    '''
    vectorizer = CountVectorizer(
        ngram_range=(3, 3), max_features=2000).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:topn]


def get_top_quadgrams(corpus, topn=20):
    '''(CountVectorizer) Most Frequently Used Quad-grams.

    EXAMPLE
    -------
    >>> top_quadgram_words = get_top_quadgrams(corpus, n=20)
    >>> top_quadgram_df = pd.DataFrame(top_quadgram_words)
    >>> top_quadgram_df.columns = ["quad-gram", "frequency"]
    >>> print(top_quadgram_df)
    '''
    vectorizer = CountVectorizer(ngram_range=(
        4, 4), max_features=2000).fit(corpus)
    bag_of_words = vectorizer.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:topn]
