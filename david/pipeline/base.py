from pandas import DataFrame, Series

from ..io.text import as_jsonl_file, as_txt_file
from ..lang import SPACY_STOP_WORDS
from ..text.prep import preprocess_sequence
from .metric import TextMetrics


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

    def clean_all_text(self, text_col="text", contractions=True,
                       lemmatize=False, punctuation=True,
                       stopwords=True, stop_words=None, tokenize=False):
        """Cleans all texts in a chained operation."""
        if not stop_words:
            stop_words = self.STOP_WORDS

        self[text_col] = self[text_col].apply(
            lambda sequence: preprocess_sequence(
                sequence, contractions, lemmatize,
                punctuation, stopwords, stop_words, tokenize))

    def custom_stopwords_from_freq(self, text_col='text',
                                   top_n=10, stop_words=None):
        """Construct a frequency stopword set."""

        common = Series(' '.join(
            self[text_col]).lower().split()).value_counts()[:top_n]
        uncommon = Series(' '.join(
            self[text_col]).lower().split()).value_counts()[-top_n:]
        if not stop_words:
            stop_words = self.STOP_WORDS
        stop_words = set(stop_words)
        stop_words = stop_words.union(list(common.keys()))
        return stop_words.union(list(uncommon.keys()))

    def slice_shape(self,
                    ref_col='stringLength',
                    min_val: int = None,
                    max_val: int = None,
                    as_copy: bool = True):
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
