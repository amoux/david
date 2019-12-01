
from gensim.corpora.textcorpus import STOPWORDS as GENSIM_STOP_WORDS
from nltk.corpus import stopwords as _nltk_stopwords
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

from .en_lexicon import DAVID_STOP_WORDS

NLTK_STOP_WORDS = _nltk_stopwords.words('english')
