import pandas as pd


class DavidDataFrameBase(pd.DataFrame):
    def __init__(self, data_structure):
        super().__init__(data_structure)


class JsonDataFrame(DavidDataFrameBase):
    JSON_PATH = None

    def __init__(self, corpus_path):
        super().__init__(pd.read_json(
            corpus_path, encoding='utf-8', lines=True))
        self.JSON_PATH = corpus_path

    def get_json_path(self):
        return self.JSON_PATH

    def to_textfile(self, fn: str, text_col='text'):
        with open(fn, 'w', encoding='utf-8') as f:
            for x in self[text_col].tolist():
                if len(x) != 0:
                    f.write('%s\n' % x)
            f.close()

    def obj_to_dict(self, df_obj: object):
        return df_obj.to_dict(orient='index')

    def missing_values(self):
        return self.isnull().sum()

    def normalize_whitespaces(self, text_col='text'):
        self[text_col] = self[text_col].str.strip()

    def slice_dataframe(self, by_min_value=0, of_column='stringLength'):
        '''Use a reference metric table to reduce the size of the dataframe.

        NOTE: This method uses a table reference obtained from either the
        `TextMetric.get_all_metrics` method or, any associated methods used
        to extract numerical information from the text sequences.

        Parameters:
        ----------

        `of_column` : (str)
            The table to use as a reference to reduce the size of a dataframe.
            You can additionally use any tables with indexed-like values. For
            instance, if the minimun value for TABLE_C is 2, then applying
            value 35; removed all rows in the corpus by that value.

        `by_min_value` : (int)
            The minimum number of values in a column to use as a rule to slice
            a dataframe.

        Does not work:
        -------------

            >>> metrics.slice_dataframe(by_min_value=40)
            >>> metrics.text.describe()
          'count 3361'
          'unique 3294'

        Works:
        -----

            >>> reduced = metrics.slice_dataframe(by_min_value=40)
            >>> reduced.text.describe()
          'count 151'
          'unique 151'
        '''
        if (by_min_value == 0):
            raise Exception('You must pass a value greater than zero!')

        return self[self[of_column] > int(by_min_value)]

    @staticmethod
    def process_fromchunks(self, infile='', sep=',', chunksize=1000,
                           outfile='df_output.csv', text_col='text',
                           standardize=True, contractions=True,
                           lemmatize=False, normalize=True,
                           rm_duplicates=True, lower_text=False):
        '''
        NOTE: This function needs to be tested and optimized.
        Chunking is importantly usefull for any large corpus so its
        not another feature. Its a must have feature here!
        For more info on this features see the notes and tests i took
        for this function on a jupyter notebook! DONT FORGET!

        Jupyter-Notebooks/David/02_pipe/chunking-testing-spacy-language-detection.ipynb
        '''

        df_chunks = pd.read_csv(infile, sep=sep, chunksize=chunksize)
        tempchunks = []
        for df in df_chunks:
            self.clean_all_text(df, text_col, standardize,
                                contractions, lemmatize, normalize,
                                rm_duplicates, lower_text)
        return pd.concat(tempchunks)
