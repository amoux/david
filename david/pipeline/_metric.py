
import emoji
import numpy as np
import pandas as pd
import spacy.lang.en
from textblob import TextBlob as _TextBlob

STOPWORDS = spacy.lang.en.STOP_WORDS


class TextMetrics(pd.DataFrame):
    SENTI_LABELS = ('positive', 'negative', 'neutral')
    JSON_PATH = None

    if not isinstance(SENTI_LABELS, tuple):
        raise TypeError('you need to pass a tuple!')

    def __init__(self, json_fpath: str):
        '''Gathers Statistical Metrics from Texts.

        Parameters
        ----------

        `file_path` : (str)
            File path where the json corpus is located/downloaded.

        * If downloading a new corpus you can alse do the following.

            >>> from david.youtube.scraper import download
            >>> from david.pipeline import TextMetrics
            ...
            >>> metric = TextMetrics(download('BmYZH7xt8sU', load_corpus=True))

        '''
        super().__init__(pd.read_json(
            json_fpath, encoding='utf-8', lines=True))
        self.JSON_PATH = json_fpath

    def to_textfile(self, fn: str, text_col='text'):
        with open(fn, 'w', encoding='utf-8') as f:
            for x in self[text_col].tolist():
                if len(x) != 0:
                    f.write('%s\n' % x)
            f.close()

    def get_json_filepath(self):
        '''Returns the path of the corpus for other instances.
        NOTE: if this works better than having to make another class
        just to swap an object from one instance to another. just
        delete this NOTE and get a coffee. Simple IS BETTER!
        '''
        return self.JSON_PATH

    def obj_todict(self, df_obj: object):
        return df_obj.to_dict(orient='index')

    def missing_values(self):
        return self.isnull().sum()

    def extract_emojis(self, str):
        emojis = ''.join(e for e in str if e in emoji.UNICODE_EMOJI)
        return emojis

    def sentiment_polarity(self, text: str):
        return _TextBlob(text).sentiment.polarity

    def sentiment_subjectivity(self, text: str):
        return _TextBlob(text).sentiment.subjectivity

    def sentiment_labeler(self, x):
        '''Labels for sentiment analysis scores.
        Add custom valus by passing to `TextMetric.SENTI_LABELS=(i,i,i)`
        '''
        if (x > 0):
            return self.SENTI_LABELS[0]
        elif (x < 0):
            return self.SENTI_LABELS[1]
        else:
            return self.SENTI_LABELS[2]

    def avgwords(self, words: list):
        return [len(w) for w in words.split(' ') if w not in STOPWORDS]

    def normalize_whitespaces(self, text_col='text'):
        self[text_col] = self[text_col].str.strip()

    def string_metric(self, text_col='text'):
        self['stringLength'] = self[text_col].str.len()

    def word_metrics(self, text_col='text'):
        '''This function applies the `spacy.lang.en.STOPWORDS`
        set to all computations; for capturing metrics of actual
        words and stopwords detected within a string.
        '''
        self['avgWordLength'] = self[text_col].apply(
            lambda x: np.mean(self.avgwords(x))
            if len(self.avgwords(x)) > 0 else 0)

        self['isStopwordCount'] = self[text_col].apply(
            lambda x: len(str(x).split()))

        self['noStopwordCount'] = self[text_col].apply(
            lambda texts: len([w for w in texts.split(' ')
                               if w not in STOPWORDS]))

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
        self['sentimentLabel'] = self['sentiPolarity'].apply(
            lambda x: self.sentiment_labeler(x))

    def extract_authortags(self, text_col='text'):
        self['authorTimeTag'] = self[text_col].str.extract(
            r'(\d{1,2}\:\d{1,2})')
        self['authorUrlLink'] = self[text_col].str.extract(r'(http\S+)')
        self['authorHashTag'] = self[text_col].str.extract(r'(\#\w+)')
        self['authorEmoji'] = self[text_col].apply(self.extract_emojis)

    def slice_dataframe(self, from_table='stringLength', by_setvalue=0):
        '''Note: For slicing a dataframe, you must call this method via
        a new variable name for slicing to work. Creating a new instance
        avoids having to restart a new session if the outcomes obtained
        after slicing were not adequate. SEE EXAMPLES BELOW!

        `from_table`: (str, default='stringLength')
            The table to use as a way to slice the dataframe.
            You can additionally use the values from 'isStopwordCount.' For
            instance, if the minimum value for `isStopwordCount=2`, then
            applying a value [10 <-> 20] to slice the dataframe; will remove
            text-rows (small sentences) which are not valuable in context.
            Recommended before modeling and text-preprocessing steps.

        `by_setvalue`: (int)
            The minimum number of values in a column to use as a rule to slice
            a dataframe.

        * Doesn't work:
        >>> metrics.slice_dataframe(by_setvalue=40)
        >>> metrics.text.describe()
          count 3361
          unique 3294

        * Works:
        >>> reduced = metrics.slice_dataframe(by_setvalue=40)
        >>> reduced.text.describe()
          count 151
          unique 151
        '''
        if by_setvalue == 0:
            raise Exception('You must pass a value greater than zero!')
        else:
            return self[self[from_table] > int(by_setvalue)]

    def get_all_metrics(self, text_col='text', string=True, words=True,
                        characters=True, sentiment=False, gettags=False):
        '''
        Single function call to get and extract
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
