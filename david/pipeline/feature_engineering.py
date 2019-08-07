import re as _re
import emoji as _emoji
import numpy as _np
import pandas as pd
from spacy.lang import en as _en
from textblob import TextBlob as _TextBlob

_STOPWORDS = _en.STOP_WORDS


class TextMetrics(pd.DataFrame):
    """Gathers Statistical Metrics from Texts.

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
                txt.write('%s\n' % line)
            txt.close()

    def missing_values(self):
        return self.isnull().sum()

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

    def avgwords(self, words: list):
        return [len(w) for w in words.split(' ') if w not in _STOPWORDS]

    def normalize_whitespaces(self, text_col='text'):
        '''Prep texts by normalizing whitespaces.'''
        self[text_col] = self[text_col].str.strip()

    def string_metric(self, text_col='text'):
        self['stringLength'] = self[text_col].str.len()

    def word_metrics(self, text_col='text'):
        '''Returns the count of words not consider as `STOPWORDS`.
        Uses the set found in Spacy's API; `spacy.lang.en.STOPWORDS`.
        avgWordLength is the average word length in a text column.
        Uses the set from Spacy's API; `spacy.lang.en.STOPWORDS`.
        '''
        self['avgWordLength'] = self[text_col].apply(
            lambda x: _np.mean(self.avgwords(x))
            if len(self.avgwords(x)) > 0 else 0)

        self['isStopwordCount'] = self[text_col].apply(
            lambda x: len(str(x).split()))

        self['noStopwordCount'] = self[text_col].apply(
            lambda texts: len([w for w in texts.split(' ')
                               if w not in _STOPWORDS]))

    def character_metrics(self, text_col='text'):
        self['charDigitCount'] = self[text_col].str.findall(
            r'[0-9]').str.len()
        self['charUpperCount'] = self[text_col].str.findall(
            r'[A-Z]').str.len()
        self['charLowerCount'] = self[text_col].str.findall(
            r'[a-z]').str.len()

    def sentiment_metrics(self, text_col='text'):
        self['sentiPolarity'] = self[text_col].apply(self.sentiment_polarity)
        self['sentiSubjectivity'] = self[text_col].apply(
            self.sentiment_subjectivity)
        self['sentimentLabel'] = self.sentiment_labeler()

    def extract_authortags(self, text_col='text'):
        self['authorTimeTag'] = self[text_col].str.extract(
            r'(\d{1,2}\:\d{1,2})')
        self['authorUrlLink'] = self[text_col].str.extract(r'(http\S+)')
        self['authorHashTag'] = self[text_col].str.extract(r'(\#\w+)')
        self['authorEmoji'] = self[text_col].apply(self.extract_emojis)

    def slice_dataframe(self, from_table='stringLength', by_min_count=0):
        '''Note: For safety, it must be returned to a new variable name in
        order for slicing to work. This avoids having to start a new instance
        if the results where not satisfactory. SEE EXAMPLES BELOW!

        `from_table`: (str, default='stringLength')
            The table to use as a way to slice the dataframe. You can also
            use the values from 'isStopwordCount'. For example, if the minimum
            value for isStopwordCount = 2, then using a value [10 <-> 20] to
            slice the dataframe; will remove text-rows (small senteces) which
            are't usefull. Recommened before modeling/text-preprocessing steps.

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
        if by_min_count == 0:
            raise Exception('You must pass a value greater than zero!')
        else:
            return self[self[from_table] > int(by_min_count)]

    def get_all_metrics(self, text_col='text', string=True, words=True,
                        characters=True, sentiment=False, gettags=False):
        '''Single function call to get and extract
        information about the text.

        METRICS
        -------
        * ( string-level metric ) if `string=True`:
            >>> `stringLength`: 'sum of all words in a string.'

        * ( word-level metrics ) if `words=True`:

            NOTE: Count/Avg values use the STOPWORDS spaCy set.

            >>> `avgWordLength`: 'average number of words.'
            >>> `isStopwordCount`: 'count of stopwords only.'
            >>> `noStopwordCount`: 'count of none stopwords.'

        * ( character-level metrics ) if `character=True`:
            >>> `charDigitCount`: 'count of digits chars.'
            >>> `charUpperCount`: 'count of uppercase chars.'
            >>> `charLowerCount`: 'count of lowercase chars.'

        * ( sentiment-level metrics ) if `sentimen=True`:
            >>> `sentiPolarity`: 'polarity score with Textblob, (float).'
            >>> `sentiSubjectivity`: 'subjectivity score with Textblob (float)'
            >>> `sentimentLabel`: 'labels row w\one (pos, neg, neutral) tag.'

        * ( tag-extraction metrics ) if `gettags=True`:
            >>> `authorTimeTag`: 'extracts video time tags, e.g. 1:20.'
            >>> `authorUrlLink`: 'extracts urls links if found.'
            >>> `authorHashTag`: 'extracts hash tags, e.g.  # numberOne
            >>> `authorEmoji`: 'extracts emojis if found.'

        '''
        self.normalize_whitespaces(text_col)
        if string:
            self.string_metric(text_col)
        if words:
            self.word_metrics(text_col)
        if characters:
            self.character_metrics(text_col)
        if sentiment:
            self.sentiment_metrics(text_col)
        if gettags:
            self.extract_authortags(text_col)

    def __repr__(self):
        return f'{ type(self) }'
