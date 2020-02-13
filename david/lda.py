import gensim
import pandas

from .ngrams import sents_to_ngramTokens
from .pipeline import Pipeline


def get_lda_main_topics(
    doc: list, num_topics: int, lda_model: object, corpus: dict,
):
    """Get the main topic per sequence using the (LDA) topic model.

    `doc` (list): An iterable of sequences or tokens sequences.
    `num_topics` (int): Number of topics key words to assing to each sequence.
    `lda_model` (object): A trained LDA model.
    `corpus` (dict): A gensim corpus dictionary from corpora.Dictionary.

    Returns: A Dataframe with dominant topics, contribution percentage and topic keywords.
    """
    df = Pipeline()
    for i, main_topic in enumerate(lda_model[corpus]):
        topics = main_topic[0] if lda_model.per_word_topics else main_topic
        topics = sorted(topics, key=lambda x: (x[1]), reverse=True)
        for j, (topic, probability) in enumerate(topics):
            if j == 0:
                keywords = ", ".join([w for w, p in lda_model.show_topic(topic)])
                contrib = round(probability, num_topics)
                topics = pandas.Series([int(topic), contrib, keywords])
                df = df.append(topics, ignore_index=True)
            else:
                break
    df = pandas.concat([df, pandas.Series(doc)], axis=1)
    df.columns = ["dominant_topic", "contribution", "keywords", "text"]
    return Pipeline(df)


def GensimLdaModel(
    doc: list,
    num_topics=10,
    random_state=100,
    update_every=0,
    chunksize=1000,
    passes=10,
    alpha="symmetric",
    iterations=50,
    per_word_topics=True,
):
    """Trains the Latent Dirichlet Allocation model.

    Loads all the required components for a LDA session:
    Returns: `lda_model, corpus, id2word`.

    `update_every` (int):
        Set to 0 for batch learning, 1 for online iterative learning.

    For more information on the model refer to the documentaion
    online: https://radimrehurek.com/gensim/models/ldamodel.html
    """
    ngram_tokens = sents_to_ngramTokens(doc)
    id2word = gensim.corpora.Dictionary(ngram_tokens)
    corpus = [id2word.doc2bow(seq) for seq in ngram_tokens]
    lda_model = gensim.models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=random_state,
        update_every=update_every,
        chunksize=chunksize,
        passes=passes,
        alpha=alpha,
        iterations=iterations,
        per_word_topics=per_word_topics,
    )
    return lda_model, corpus, id2word
