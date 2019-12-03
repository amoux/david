from pandas import DataFrame, Series

from ..io.text import as_jsonl_file, as_txt_file
from ..lang import SPACY_STOP_WORDS
from ..text.prep import preprocess_sequence, normalize_wiggles
from .metric import TextMetrics

TIME_RE = r"(\d{1,2}\:\d{1,2})"
URL_RE = r"(http\S+)"
TAG_RE = r"(\#\w+)"


class DataFrameBase(DataFrame):
    def __init__(self, *args, **kwargs):
        super(DataFrameBase, self).__init__(*args, **kwargs)


class Pipeline(DataFrameBase, TextMetrics):
    STOP_WORDS = SPACY_STOP_WORDS

    @property
    def to_dict_obj(self):
        return self.to_dict(orient='index')

    def to_text_file(self, fname: str, output_dir='.', text_col='text'):
        texts = self[text_col].values.tolist()
        as_txt_file(texts, fname, output_dir)

    def to_jsonl_file(self, fname: str, output_dir='.', text_col='text'):
        texts = self[text_col].values.tolist()
        as_jsonl_file(texts, fname, output_dir)

    def replace_authortags(self, text_col='text'):
        self[text_col] = self[text_col].str.replace(TIME_RE, " ")
        self[text_col] = self[text_col].str.replace(URL_RE, " ")
        self[text_col] = self[text_col].str.replace(TAG_RE, " ")

    def clean_sequences(
        self,
        contractions=True,
        lemmatize=False,
        punctuation=True,
        stopwords=True,
        stop_words=None,
        tokenize=False,
        tags=False,
        wiggles=False,
        text_col="text",
    ):
        """Cleans all texts in a chained operation."""
        if tags:
            self.replace_authortags(text_col=text_col)
        if wiggles:
            self[text_col] = self[text_col].apply(
                lambda x: normalize_wiggles(x)
            )
        stop_words = stop_words if stop_words else self.STOP_WORDS
        self[text_col] = self[text_col].apply(
            lambda sequence: preprocess_sequence(
                sequence, contractions, lemmatize,
                punctuation, stopwords, stop_words, tokenize))

    def get_most_frequent_words(
        self,
        text_col='text',
        top_num=10,
        stop_words=None,
    ):
        """Construct a frequency word collection from top negative and
        positive words found across all texts.

        Parameters:
        ----------

        stop_words (Type[set, list], default=Pipeline.STOP_WORDS):
            The number of most frequent words found in all texts are added
            to the default Pipeline.STOP_WORDS - if the argument is left as
            None. If you want to only get the most most frequent words found,
            then simply pass an empty dict object to the stop_word argument.

        """
        common = Series(' '.join(
            self[text_col]).lower().split()).value_counts()[:top_num]
        uncommon = Series(' '.join(
            self[text_col]).lower().split()).value_counts()[-top_num:]
        stop_words = set(stop_words if stop_words else self.STOP_WORDS)
        stop_words = stop_words.union(list(common.keys()))
        return stop_words.union(list(uncommon.keys()))

    def slice_shape(
            self,
            ref_col='stringLength',
            min_val: int = None,
            max_val: int = None,
            as_copy=True,
    ):
        """Use a reference metric table to reduce the size of the dataframe.

        Usage:

            >>> pipe_copy = pipe.slice_shape('stringLength', min_val=40)
            >>> pipe_copy.text.describe()
            >>> 'count 151'
            >>> 'unique 151'
        """
        temp_df = self.copy(deep=as_copy)
        if min_val:
            temp_df = temp_df.loc[temp_df[ref_col] > int(min_val)]
        if max_val:
            temp_df = temp_df.loc[temp_df[ref_col] < int(max_val)]
        return Pipeline(temp_df)
