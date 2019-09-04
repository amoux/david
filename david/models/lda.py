
import pandas as pd

LDA_TOPIC_COLNAMES = ['dominant_topic', 'contribution(%)', 'keywords']

NEW_TOPIC_COLNAMES = ['topicId', 'dominantTopic',
                      'topicContribution', 'topicKeywords', 'topicTokens']


def build_topics(LDA_model, Gensim_doc2bow,
                 corpus: str, num_topics: int,
                 reset_col_names: bool = False,
                 topic_col_names: list = NEW_TOPIC_COLNAMES
                 ):
    '''(LDA) Gets the dominant topic, percentage contribution and
    the keywords for each given text document. This function identifies
    a topic related to the given text documents and flags all text items
    with high probability for a keyword topic.

    NOTE: Pass a list containing five names for the topic column names to
    override the defaults in order and set `reset_col_names=True`:

    `NEW_TOPIC_COLNAMES` = [
        'topicId', 'dominantTopic', 'topicContribution',
        'topicKeywords', 'topicTokens'
        ]

    Parameters:
    ----------

    `LDA_model` : (object)
        A trained LDA (Latent Dirichlet Allocation) model.

    `Gensim_doc2bow` : (object)
        A gensim dictionary object from the corpora.Dictionary
        module. NOTE: The dict must be passed to gensim before
        passing it to this function.

    `corpus` : (list[str])
        The text column in a pandas.Dataframe containing texts.
        The text must be preprocessed and vectorized before using
        this function.

    `num_topics` : (int)
        Number of topics keywords to assing to each
        text sentence in the corpus

    Returns:
    -------
    Returns a pandas.Dataframe containing tokenized topic text sentences
    and the document's dominant (top) keywords.

    '''
    if not num_topics:
        raise ValueError('You need to pass a value for the number of topics!')

    df = pd.DataFrame()
    # get main topic in each document.
    for _, doc in enumerate(LDA_model[Gensim_doc2bow]):
        doc_row = doc[0] if LDA_model.per_word_topics else doc
        doc_row = sorted(doc_row, key=lambda x: (x[1]), reverse=True)
        # get the top dominant topics.
        for idx, (topic, topic_probability) in enumerate(doc_row):
            if idx == 0:
                # get a representation for selected topics.
                topic_terms = LDA_model.show_topic(topic)
                topic_keywords = ", ".join([w for w, prop in topic_terms])
                df = df.append(pd.Series(
                    [
                        int(topic),
                        round(topic_probability, num_topics),
                        topic_keywords
                    ]
                ), ignore_index=True)
            else:
                break

    df.columns = LDA_TOPIC_COLNAMES
    # add original text to the end of the output.
    df_contents = pd.Series(corpus)
    df = pd.concat([df, df_contents], axis=1)

    if reset_col_names and len(topic_col_names) == 5:
        NEW_TOPIC_COLNAMES = topic_col_names
        df = df.reset_index()
        df.columns = NEW_TOPIC_COLNAMES
        return df
    else:
        return df
