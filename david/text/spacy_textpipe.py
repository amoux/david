import spacy

en_core_web_lg = spacy.load('en_core_web_lg')


def spacy_sent_tokenizer(docs: list, nlp_model: object = None):
    """Spacy sentence tokenizer

    NOTE: sentence tokenizer requires the large english model.

    Parameters:
    -----------
    docs : (list)
        An iterable list containing texts.
    nlp_model : (object)
        A spacy language model instance to use with the sentence tokenizer.
    """
    if not nlp_model:
        nlp_model = en_core_web_lg
    sent_tokens = []
    for doc in docs:
        doc = nlp_model(doc)
        for token in doc.sents:
            sent_tokens.append(token.text)
    return sent_tokens
