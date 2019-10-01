from pandas import DataFrame


class DavidDataFrame(DataFrame):

    def __init__(self, *args, **kwargs):
        super(DavidDataFrame, self).__init__(*args, **kwargs)

    @property
    def obj_to_dict(self):
        return self.to_dict(orient='index')

    @property
    def missing_values(self):
        return self.isnull().sum()

    def to_textfile(self, fn: str, text_col='text') -> None:
        '''Writes string sequences to a text file.
        '''
        with open(fn, 'w', encoding='utf-8') as f:
            texts = self[text_col].values.tolist()
            for text in texts:
                if len(text) > 0:
                    f.write('%s\n' % text)
            f.close()

    def slice_shape(self,
                    ref_col: str = 'stringLength',
                    min_val: int = None,
                    max_val: int = None,
                    as_copy: bool = True):
        '''Use a reference metric table to reduce the size of the dataframe.

        Parameters:
        ----------

        `ref_col` : (type=str)
            The table to use as a reference to reduce the size of a dataframe.
            You can additionally use any tables with indexed-like values. For
            instance, if the minimun value for TABLE_C is 2, then applying
            value 35; removed all rows in the corpus by that value.

        `min_val` : (type=int)
            The minimum number of values in a column to use as a rule to slice
            a dataframe.

        Does not work:
        -------------

            >>> metrics.slice_shape(min_val=40)
            >>> metrics.text.describe()
          'count 3361'
          'unique 3294'

        Works:
        -----

            >>> reduced = metrics.slice_shape(min_val=40)
            >>> reduced.text.describe()
          'count 151'
          'unique 151'

        '''
        temp_df = self.copy(deep=as_copy)
        if min_val > 0:
            temp_df = temp_df.loc[temp_df[ref_col] > int(min_val)]
        if max_val > 0:
            temp_df = temp_df.loc[temp_df[ref_col] < int(max_val)]
        return temp_df
