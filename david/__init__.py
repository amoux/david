"""Main logging for david.

David uses Python's default logging system.
Each module has it's own logger, so you can control the verbosity of each
module to your needs. To change the overall log level, you can set log levels
at each part of the module hierarchy, including simply at the root, `david`:
```ipython
import logging
logging.getLogger('david').setLevel(logging.INFO)
```
Logger-Modes:
    DEBUG   : information typically of interest only when diagnosing problems.
    INFO    : confirmation that things are working as expected.
    WARN    : indication that something unexpected happened.
    ERROR   : software has not been able to perform some function.
    CRITICAL: program itself may be unable to continue running.
"""

import logging
import warnings

from david.cosine import (build_feature_matrix, cosine_similarity,
                          get_similar_docs)
from david.datasets import JsonlYTDatasets
from david.io import (File, GoogleDriveDownloader, as_jsonl_file, as_txt_file,
                      delete_files, download_url_file)
from david.lang import (DAVID_STOP_WORDS, GENSIM_STOP_WORDS, SPACY_STOP_WORDS,
                        SpellCorrect, TextSearchContractions)
from david.lda import GensimLdaModel, get_lda_main_topics
from david.models import elmo as ElmoModel
from david.models import trans as TransformersModel
from david.ngrams import (sents_to_ngramTokens, top_bigrams, top_quadgrams,
                          top_trigrams, top_unigrams)
from david.pipeline import Pipeline
from david.server import CommentsSQL
from david.text.prep import (YTCommentTokenizer, encode_ascii,
                             nltk_word_tokenizer, normalize_whitespace,
                             normalize_wiggles, part_of_speech_annotator,
                             part_of_speech_lemmatizer, preprocess_sequence,
                             remove_punctuation, remove_repeating_characters,
                             remove_repeating_words, remove_stopwords,
                             treebank_to_wordnet_pos)
from david.text.viz import build_wordcloud
from david.youtube import (YTCommentScraper, YTRegexMatchers, YTSpecialKeys,
                           yt_query_search, yt_query_video_content,
                           yt_query_videoids)

logging.basicConfig(level=logging.INFO)
del logging

# silence tensorflow warnings.
warnings.filterwarnings('ignore', category=DeprecationWarning, module='google')
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="DeprecationWarning")


# seeds random states for sources.
seed = 0

# david version - setup.py imports this value
__version__ = '0.0.2'
