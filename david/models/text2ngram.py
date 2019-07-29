
import gensim
import spacy
from gensim.utils import simple_preprocess
from spacy.lang.en.stop_words import STOP_WORDS


POS_TAGS = ['NOUN', 'ADJ', 'VERB', 'ADV']
MODEL = 'en_core_web_lg'


def process_words(
    corpus,
    model=MODEL,
    stop_words=STOP_WORDS,
    allowed_postags=POS_TAGS
):
    """Remove stopwords, form bigrams, trigrams and Lemmatization
    This function uses a list of STOP_WORDS list from spaCy's API.
    Additionally, it uses the gensim.utils.simple_preprocess module,
    which: Converts a document into a list of lowercase tokens,
    ignoring tokens that are too short or too long.

    PARAMETERS
    ----------
    corpus : text dataset
        A processed text data with: data_ready = process_words(corpus)

    model : (spacy language model, default='en_core_web_lg')
        Other spacy models are compatible with this function.

    RETURNS:
        Returns preprocessed texts.
    """
    # build biagrams and trigram models
    bigram = gensim.models.Phrases(corpus, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[corpus], threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    texts = [[word for word in simple_preprocess(str(doc))
              if word not in stop_words] for doc in corpus]

    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]

    texts_out = []
    nlp = spacy.load(model, disable=['parser', 'ner'])

    spacy.prefer_gpu()
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc
                          if token.pos_ in allowed_postags])

    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc))
                  if word not in stop_words] for doc in texts_out]
    return texts_out
