
import pandas as pd


def format_sent_topics(LDA_model, gensim_dict, corpus, num_topics: int):
    """(LDA) Gets the dominant topic, percentage contribution and
    the keywords for each given text document. This function identifies
    a topic related to the given text documents and flags all text items
    with high probability for a keyword topic.

    PARAMETERS
    ----------
    LDA_model : (object)
        A trained LDA (Latent Dirichlet Allocation) model.

    gensim_dict : (object)
        A gensim dictionary object from the corpora.Dictionary
        module. NOTE: The dict must be passed to gensim before
        passing it to this function.

    corpus : (df)
        The text column in a pandas.Dataframe containing texts.
        The text must be preprocessed and vectorized before using
        this function.

    num_topics : (int)
        Number of topics keywords to assing to each
        text sentence in the corpus

    Returns
    -------
        Returns a pandas.Dataframe containing tokenized topic text sentences
        and the document's dominant (top) keywords.
    """
    # dataframe instance for the topics
    df_topics = pd.DataFrame()
    # get main topic in each document
    for i, doc in enumerate(LDA_model[gensim_dict]):
        doc_row = doc[0] if LDA_model.per_word_topics else doc
        doc_row = sorted(doc_row, key=lambda x: (x[1]), reverse=True)
        # get the top dominant topics
        for topic, (topic_idx, topic_probability) in enumerate(doc_row):
            if topic == 0:
                # get a representation for selected topics
                topic_terms = LDA_model.show_topic(topic_idx)
                topic_keywords = ", ".join([w for w, prop in topic_terms])
                df_topics = df_topics.append(pd.Series([
                    int(topic_idx),
                    round(topic_probability, num_topics), topic_keywords]
                ), ignore_index=True)
            else:
                break
    # column names for the topic dataframe instance
    df_topics.columns = ['dominant_topic', 'contribution(%)', 'keywords']
    # add original text to the end of the output
    df_contents = pd.Series(corpus)
    df_topics = pd.concat([df_topics, df_contents], axis=1)
    return(df_topics)
