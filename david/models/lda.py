
import pandas as pd

TOPIC_COLUMN_NAMES = ['dominant_topic', 'contribution(%)', 'keywords']


def get_lda_topics(LDA_model, Gensim_dict,
                   text_col: str, num_topics: int,
                   topic_col_names: list = TOPIC_COLUMN_NAMES
                   ):
    '''(LDA) Gets the dominant topic, percentage contribution and
    the keywords for each given text document. This function identifies
    a topic related to the given text documents and flags all text items
    with high probability for a keyword topic.

    NOTE: Pass a list containing three names for the topic column names to
    override the defaults in the following order:

    `TOPIC_COLUMN_NAMES=['dominant_topic', 'contribution(%)', 'keywords']`

    Parameters:
    ----------

    `LDA_model` : (object)
        A trained LDA (Latent Dirichlet Allocation) model.

    `Gensim_dict` : (object)
        A gensim dictionary object from the corpora.Dictionary
        module. NOTE: The dict must be passed to gensim before
        passing it to this function.

    `text_col` : (list[str])
        The text column in a pandas.Dataframe containing texts.
        The text must be preprocessed and vectorized before using
        this function.

    `num_topics` : (int)
        Number of topics keywords to assing to each
        text sentence in the text_col

    Returns:
    -------
        Returns a pandas.Dataframe containing tokenized topic text sentences
        and the document's dominant (top) keywords.

    '''
    if topic_col_names and len(topic_col_names) == 3:
        TOPIC_COLUMN_NAMES = topic_col_names

    if not num_topics:
        raise ValueError('You need to pass a value for the number of topics!')

    # dataframe instance for the topics.
    df = pd.DataFrame()

    # get main topic in each document.
    for _, doc in enumerate(LDA_model[Gensim_dict]):
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

    # assign the column names for the topic dataframe instance.
    df.columns = TOPIC_COLUMN_NAMES
    # add original text to the end of the output.
    df_contents = pd.Series(text_col)
    df = pd.concat([df, df_contents], axis=1)
    return (df)
