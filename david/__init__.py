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

from david.youtube import (YTCommentScraper, YTRegexMatchers, YTSpecialKeys,
                           yt_query_search, yt_query_video_content,
                           yt_query_videoids)
from david.visualizers import build_wordcloud
from david.utils.main import get_data_home, is_cuda_enabled, remove_data_home
from david.utils.loaders import (GoogleDriveDownloader, current_path,
                                 delete_files, pointer)
from david.utils.io import as_jsonl_file, as_txt_file
from david.text.spacy_textpipe import sent_tokenizer
from david.text.nltk_textpipe import (encode_ascii, expand_contractions,
                                      nltk_tokenizer, preprocess_doc,
                                      preprocess_docs,
                                      remove_repeated_characters,
                                      remove_stopwords,
                                      treebank_to_wordnet_postag,
                                      wordnet_lemmatizer)
from david.text._prep_v1 import (get_emojis, get_sentiment_polarity,
                                 get_sentiment_subjectivity, get_vocab_size,
                                 lemmatizer, reduce_repeating_chars_v1,
                                 regex_tokenizer, remove_duplicate_words)
from david.server import CommentsSQL
from david.pipeline import Pipeline
from david.lda import build_topics
from david.lang import (DAVID_CONTRACTIONS, DAVID_STOP_WORDS,
                        GENSIM_STOP_WORDS, SPACY_STOP_WORDS, SpellCorrect)
from david.cosine import (CountVectorizer, TfidfVectorizer,
                          build_feature_matrix, cosine_similarity,
                          get_similar_docs)
import logging
import warnings
logging.basicConfig(level=logging.WARN)
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
