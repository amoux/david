
import pandas as pd

LDA_TOPIC_COLNAMES = ['dominant_topic', 'contribution(%)', 'keywords']

TOPIC_COLNAMES = [
    'topicId', 'dominantTopic', 'topicContribution',
    'topicKeywords', 'topictTokens']


def build_topics(
        lda_model: object,
        gensim_doc2bow: dict,
        corpus: list,
        n_topics: int,
        reset_col_names: bool = False,
        topic_col_names: list = TOPIC_COLNAMES
):
    '''Latent Dirichlet Allocation (LDA) Topic Modeling.

    Gets the dominant topic, percentage contribution and the
    keywords for each given text document. It flags all text
    items with high probability for a keyword topic.

    Parameters:
    ----------

    `lda_model` : (object)
        A trained LDA model.

    `gensim_doc2bow` : (dict)
        A gensim dictionary object from corpora.Dictionary
        module. NOTE: The dict must be passed to gensim before
        passing it to this method.

    `corpus` : (list)
        The column in a DataFrame containing the texts.

    `n_topics` : (int)
        Number of topics keywords to assing to each text
        sentence in the corpus.

    `topic_col_names` : (list, default=TOPIC_COLNAMES)
        default column names: topicId, dominantTopic,
        topicContribution, topicKeywords, topicTokens

    Returns:
    -------
    Returns a pandas.Dataframe containing tokenized
    topic text sentences and the document's dominant
    (top) keywords.
    '''
    if not n_topics:
        raise ValueError('You need to pass a value for the number of topics!')

    df = pd.DataFrame()
    for _, doc in enumerate(lda_model[gensim_doc2bow]):
        doc = doc[0] if lda_model.per_word_topics else doc
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        for i, (topic, probability) in enumerate(doc):

            if i == 0:
                topics = lda_model.show_topic(topic)
                keywords = ', '.join([w for w, p in topics])
                df = df.append(
                    pd.Series(
                        [int(topic), round(probability, n_topics),
                         keywords]), ignore_index=True)
            else:
                break

    df.columns = LDA_TOPIC_COLNAMES
    corpus = pd.Series(corpus)
    df = pd.concat([df, corpus], axis=1)

    if reset_col_names and len(topic_col_names) == 5:
        TOPIC_COLNAMES = topic_col_names
        df = df.reset_index()
        df.columns = TOPIC_COLNAMES
        return df
    else:
        return df
