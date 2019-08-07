import re as _re
import emoji as _emoji
import numpy as _np
import pandas as pd
from spacy.lang import en as _en
from textblob import TextBlob as _TextBlob

_STOPWORDS = _en.STOP_WORDS


class TextMetrics(pd.DataFrame):
    """Gathers statistical information from text.
    "A Pandas subclass which initializes the collection of functions
    only work by passing a dataframe object.

    `file_path` (str):
        file path passed into pd.read_json().
    """

    def __init__(self, file_path: str):
        super().__init__(pd.read_json(file_path, encoding='utf-8', lines=True))

    def to_textfile(self, filename, text_column='text'):
        '''Save the texts from a column to a text file.
        '''
        with open(filename, 'w', encoding='utf-8') as txt:
            lines = self[text_column].tolist()
            for line in lines:
                txt.write('%s\n' & line)
            txt.close()

    def missing_values(self):
        return self.isnull().sum()

    def prep_textcolumn(self, text_col='text'):
        '''Prep texts by normalizing whitespaces.
        '''
        self['text'] = self[text_col].str.strip()

    def extract_emojis(self, str):
        emojis = ''.join(e for e in str if e in _emoji.UNICODE_EMOJI)
        return emojis

    def sentiment_polarity(self, text: str):
        return _TextBlob(text).sentiment.polarity

    def sentiment_subjectivity(self, text: str):
        return _TextBlob(text).sentiment.subjectivity

    def sentiment_labeler(self):
        labels = []
        for score in self['sentiPolarity']:
            if (score > 0):
                labels.append("positive")
            elif (score < 0):
                labels.append("negative")
            elif (score == 0):
                labels.append("neutral")
        return labels

    def count_none_stopwords(self, text_col='text'):
        '''Returns the count of words not consider as `STOPWORDS`.
        Uses the set found in Spacy's API; `spacy.lang.en.STOPWORDS`.
        '''
        self['notStopwordsCount'] = self[text_col].apply(
            lambda texts: len([w for w in texts.split(' ')
                               if w not in _STOPWORDS]))

    def _avgwords(self, words: list):
        return [len(w) for w in words.split(' ') if w not in _STOPWORDS]

    def word_mean_average(self, text_col='text'):
        '''Returns the average word length in a text column.
        Uses the set from Spacy's API; `spacy.lang.en.STOPWORDS`.
        '''
        self['wordAvgLength'] = self[text_col].apply(
            lambda x: _np.mean(self._avgwords(x))
            if len(self._avgwords(x)) > 0 else 0)

    def extract_textmetrics(self, text_col='text'):
        self['wordStrLen'] = self[text_col].str.len()
        self['charIsDigitCount'] = self[text_col].str.findall(
            r'[0-9]').str.len()
        self['charIsUpperCount'] = self[text_col].str.findall(
            r'[A-Z]').str.len()
        self['charIsLowerCount'] = self[text_col].str.findall(
            r'[a-z]').str.len()
        self['hasStopwordsCount'] = self[text_col].apply(
            lambda x: len(str(x).split()))

    def extract_authortags(self, text_col='text'):
        self['authorTimeTag'] = self[text_col].str.extract(
            r'(\d{1,2}\:\d{1,2})')
        self['authorUrlLink'] = self[text_col].str.extract(r'(http\S+)')
        self['authorHashTag'] = self[text_col].str.extract(r'(\#\w+)')
        self['authorEmoji'] = self[text_col].apply(self.extract_emojis)

    def sentiment_fromtexts(self, text_col='text'):
        self['sentiPolarity'] = self[text_col].apply(self.sentiment_polarity)
        self['sentiSubjectivity'] = self[text_col].apply(
            self.sentiment_subjectivity)
        self['sentimentLabel'] = self.sentiment_labeler()

    def slice_dataframe(self, from_table='notStopwordsCount', by_min_count=0):
        '''Note: For safety, it must be returned to a new variable name in
        order for slicing to work. This avoids having to start a new instance
        if the results where not satisfactory. SEE EXAMPLES BELOW!

        `from_table`: (str, default='notStopwordsCount')
            The table to use as a way to slice the dataframe.

        `by_min_count`: (int)
            The minimum number of values in a column to use as
            a rule to slice a dataframe.

        * Doesn't work:
            >>> metrics.slice_dataframe(by_min_count=40)
            >>> metrics.text.describe()
            count 3361
            unique 3294

        * Works:
            >>> reduced = metrics.slice_dataframe(by_min_count=40)
            >>> reduced.text.describe()
            count 151
            unique 151
        '''
        if not by_min_count > 0:
            raise Exception('You must pass a value greater than zero!')
        else:
            return self[self[from_table] > int(by_min_count)]

    def get_all_metrics(self, text_col='text', word_mean_avg=True,
                        none_stopwords=True, gettags=False, sentiment=False):
        '''Single function call to get and extract information about the text.

        METRICS
        -------

        * ( standard defaults ):
            >>> `wordStrLen`: 'the sum of words (length) in the text.'
            >>> `charIsDigitCount`: 'count of digits chars.'
            >>> `charIsUpperCount`: 'count of uppercase chars.'
            >>> `charIsLowerCount`: 'count of lowercase chars.'
            >>> `hasStopwordsCount`: 'count stopwords in a text.'

        * ( uses spaCy's STOPWORDS set ) if `word_mean_ag` = True:
            >>> `wordAvgLength`: 'average word length in a text column.'
            NOTE: Average values do not include STOPWORDS.

        * ( uses spaCy's STOPWORDS set ) if `none_stopwords` = True:
            >>> `notStopwordsCount`: 'counts words excluding stopwords.'
            NOTE: Count values do not include STOPWORDS.

        * ( optional-metrics ) if `gettags` = `True`:
            >>> `authorTimeTag`: 'extracts video time tags, e.g. 1:20.'
            >>> `authorUrlLink`: 'extracts urls links if found.'
            >>> `authorHashTag`: 'extracts hash tags, e.g.  # numberOne
            >>> `authorEmoji`: 'extracts emojis if found.'

        * ( optional-metrics ) if `sentiment` = `True`:
            >>> `sentiPolarity`: 'polarity score with Textblob, (float).'
            >>> `sentiSubjectivity`: 'subjectivity score with Textblob (float)'
            >>> `sentimentLabel`: 'labels row w\one (pos, neg, neutral) tag.'

        '''
        self.prep_textcolumn(text_col)
        self.extract_textmetrics(text_col)
        if word_mean_avg:
            self.word_mean_average(text_col)
        if none_stopwords:
            self.count_none_stopwords(text_col)
        if gettags:
            self.extract_authortags(text_col)
        if sentiment:
            self.sentiment_fromtexts(text_col)

    def __repr__(self):
        return f'{ type(self) }'
