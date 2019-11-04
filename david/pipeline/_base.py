from pandas import DataFrame, Series

from ..lang import SPACY_STOP_WORDS
from ..utils.io import as_jsonl_file as _as_jsonl_file
from ..utils.io import as_txt_file as _as_txt_file


class DavidDataFrame(DataFrame):
    STOP_WORDS = SPACY_STOP_WORDS

    def __init__(self, *args, **kwargs):
        super(DavidDataFrame, self).__init__(*args, **kwargs)

    @property
    def obj_to_dict(self):
        return self.to_dict(orient='index')

    @property
    def missing_values(self):
        return self.isnull().sum()

    def as_txt_file(self, fname: str, output_dir='.', text_col='text'):
        texts = self[text_col].values.tolist()
        _as_txt_file(texts, fname, output_dir)

    def as_jsonl_file(self, fname: str, output_dir='.', text_col='text'):
        texts = self[text_col].values.tolist()
        _as_jsonl_file(texts, fname, output_dir)

    def custom_stopwords_from_freq(self, text_col='text',
                                   top_n=10, stop_words=None):
        """Construct a new custom stop-word set from the top most frequently
        used words in the corpus.

        Returns a new set of from the frequency of words in the corpus and
        the existing stop word collection.

        Parameters:
        -----------
        top_n : (int)
            The number of top words to include to an existent stop word
            collection. e.g., top_n=10  counts words from both sides of
            the corpus - most frequent negative and positive words.
        stop_words : ([set|list])
            An existing collection of stop words to use and include frequent
            words from the corpus. If left as None, spaCy's stop word set is
            used instead.
        """
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

        Parameters:
        ----------
        ref_col : (str)
            The table to use as a reference to reduce the size of a dataframe.
            You can additionally use any tables with indexed-like values. For
            instance, if the minimun value for TABLE_C is 2, then applying
            value 35; removed all rows in the corpus by that value.
        min_val : (int)
            The minimum number of values in a column to use as a rule to slice
            a dataframe.

        Usage:
        -----
            >>> pipe = Pipeline()
            >>> pipe.get_all_metrics(string=True)
            >>> pipe.text.describe()
            >>> 'count 3361'
            >>> 'unique 3294'
            >>> pipe_copy = pipe.slice_shape('stringLength', min_val=40)
            >>> pipe_copy.text.describe()
            >>> 'count 151'
            >>> 'unique 151'
        """
        temp_df = self.copy(deep=as_copy)
        if min_val > 0:
            temp_df = temp_df.loc[temp_df[ref_col] > int(min_val)]
        if max_val > 0:
            temp_df = temp_df.loc[temp_df[ref_col] < int(max_val)]
        return temp_df
