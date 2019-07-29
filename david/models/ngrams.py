from sklearn.feature_extraction.text import CountVectorizer


def get_top_unigrams(corpus, topn=5):
    '''`CountVectorizer` Most Frequently Occurring `Unigrams`.

        >>> top_words = get_top_unigrams(corpus, n=5)
        >>> top_df = pd.DataFrame(top_words)
        >>> top_df.columns = ["unigram", "frequency"]
    '''
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=True)
    return wordfreq[:topn]


def get_top_bigrams(corpus, topn=5):
    '''`CountVectorizer` Most Frequently Occurring `Bi-grams`.

    Convert the most frequently bigram words to a df
    for example to plot words to a bar plot:

        >>> top_bigram_words = get_top_bigrams(corpus, n=5)
        >>> top_bigram_df = pd.DataFrame(top_bigram_words)
        >>> top_bigram_df.columns = ["bi-gram", "frequency"]
        >>> print(top_bigram_df)
    '''
    vec = CountVectorizer(ngram_range=(2, 2), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=True)
    return wordfreq[:topn]


def get_top_trigrams(corpus, topn=5):
    '''`CountVectorizer` Most Frequently Used `Tri-grams`.

        >>> top_triagram_words = get_top_trigrams(corpus, n=5)
        >>> top_triagram_df = pd.DataFrame(top_triagram_words)
        >>> top_triagram_df.columns = ["tri-gram", "frequency"]
        >>> print(top_triagram_df)
    '''
    vec = CountVectorizer(ngram_range=(3, 3), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=True)
    return wordfreq[:topn]


def get_top_quadgrams(corpus, topn=5):
    '''`CountVectorizer` Most Frequently Used `Quad-grams`.

        >>> top_quadgram_words = get_top_quadgrams(corpus, n=5)
        >>> top_quadgram_df = pd.DataFrame(top_quadgram_words)
        >>> top_quadgram_df.columns = ["quad-gram", "frequency"]
        >>> print(top_quadgram_df)
    '''
    vec = CountVectorizer(ngram_range=(4, 4), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    wordsums = bag_of_words.sum(axis=0)
    wordfreq = [(word, wordsums[0, idx])
                for word, idx in vec.vocabulary_.items()]
    wordfreq = sorted(wordfreq, key=lambda x: x[1], reverse=True)
    return wordfreq[:topn]
