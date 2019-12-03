import gensim
import pandas

from .ngrams import sents_to_ngramTokens


def build_topics(lda_model, doc2bow, corpus, n_topics):
    """Latent Dirichlet Allocation (LDA) Topic Modeling.

    Parameters:
    ----------

    `lda_model` (object): A trained LDA model.

    `doc2bow` (Type[Dict]):
        A gensim dictionary object from corpora.Dictionary module.

    `corpus` (Type[List]):
        The column in a DataFrame containing the texts.

    `n_topics` (Type[int]):
        Number of topics keywords to assing to each text sentence in
        the corpus.

    Returns: A Dataframe containing tokenized topic text
        sentences and the document's dominant (top) keywords.

    """
    df = pandas.DataFrame()
    for _, doc in enumerate(lda_model[doc2bow]):
        doc = doc[0] if lda_model.per_word_topics else doc
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        for i, (topic, probability) in enumerate(doc):
            if i == 0:
                topics = lda_model.show_topic(topic)
                keywords = ", ".join([w for w, p in topics])
                df = df.append(pandas.Series([
                    int(topic), round(probability, n_topics), keywords
                ]), ignore_index=True)
            else:
                break
    df.columns = ["dominant_topic", "contribution(%)", "keywords"]
    corpus = pandas.Series(corpus)
    return pandas.concat([df, corpus], axis=1)


def GensimLdaModel(
    doc: list,
    num_topics=10,
    random_state=100,
    upandasate_every=1,
    chunksize=1000,
    passes=10,
    alpha='symmetric',
    iterations=50,
    per_word_topics=True
):
    """Train and use Online Latent Dirichlet Allocation.

    Loads all the required components for a LDA session:
    Returns: `lda_model, corpus, id2word`
    """
    ngram_tokens = sents_to_ngramTokens(doc)
    id2word = gensim.corpora.Dictionary(ngram_tokens)
    corpus = [id2word.doc2bow(seq) for seq in ngram_tokens]
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=random_state,
        upandasate_every=upandasate_every,
        chunksize=chunksize,
        passes=passes,
        alpha=alpha,
        iterations=iterations,
        per_word_topics=per_word_topics)
    return lda_model, corpus, id2word
