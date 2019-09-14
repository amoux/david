
from collections import MutableSequence

import numpy as np
from spacy.lang.en import STOP_WORDS

from .text import (get_emojis, get_sentiment_polarity,
                   get_sentiment_subjectivity)


class TextMetrics(MutableSequence, object):
    '''Gathers Statistical Metrics from Texts.

    Parameters:
    ----------

    `file_path` : (str)
        File path where the json corpus is located.

    * If downloading a new corpus you can alse do the following.

        >>> from david.youtube.scraper import download
        >>> from david.pipeline import TextMetrics
        ...
        >>> metric = TextMetrics(download('BmYZH7xt8sU', load_corpus=True))

    '''
    STOPWORDS = STOP_WORDS
    SENTI_LABELS = ('positive', 'negative', 'neutral')

    if not isinstance(SENTI_LABELS, tuple):
        raise ValueError(f'SENTI_LABELS has to be a: {type(tuple)}.')

    if not isinstance(STOPWORDS, set):
        raise ValueError(f'STOPWORDS has to be a: {type(set)}.')

    def strip_spaces(self, text_col='text'):
        self[text_col] = self[text_col].str.strip()

    def sentiment_labeler(self, score):
        '''Labels for sentiment analysis scores.
        Add custom valus by passing to `TextMetric.SENTI_LABELS=(i,i,i)`
        '''
        if (score > 0):
            return self.SENTI_LABELS[0]
        if (score < 0):
            return self.SENTI_LABELS[1]
        else:
            return self.SENTI_LABELS[2]

    def avg_words(self, words: list):
        return [len(w) for w in words.split(' ') if w not in self.STOPWORDS]

    def string_metric(self, text_col='text'):
        self['stringLength'] = self[text_col].str.len()

    def word_metrics(self, text_col='text'):
        '''Applies the `spacy.lang.en.STOP_WORDS` set to all
        computations; captures metrics of actual words
        and stopwords detected within a string sequence.
        '''
        self['avgWordLength'] = self[text_col].apply(
            lambda x: np.mean(self.avg_words(x))
            if len(self.avg_words(x)) > 0 else 0)

        self['isStopwordCount'] = self[text_col].apply(
            lambda x: len(str(x).split()))

        self['noStopwordCount'] = self[text_col].apply(
            lambda texts: len([w for w in texts.split(' ')
                               if w not in self.STOPWORDS]))

    def character_metrics(self, text_col='text'):
        self['charDigitCount'] = self[text_col].str.findall(
            r'[0-9]').str.len()
        self['charUpperCount'] = self[text_col].str.findall(
            r'[A-Z]').str.len()
        self['charLowerCount'] = self[text_col].str.findall(
            r'[a-z]').str.len()

    def sentiment_metrics(self, text_col='text'):
        self['sentiPolarity'] = self[text_col].apply(get_sentiment_polarity)
        self['sentiSubjectivity'] = self[text_col].apply(
            get_sentiment_subjectivity)
        self['sentimentLabel'] = self['sentiPolarity'].apply(
            lambda x: self.sentiment_labeler(x))

    def extract_authortags(self, text_col='text'):
        self['authorTimeTag'] = self[text_col].str.extract(
            r'(\d{1,2}\:\d{1,2})')
        self['authorUrlLink'] = self[text_col].str.extract(r'(http\S+)')
        self['authorHashTag'] = self[text_col].str.extract(r'(\#\w+)')
        self['authorEmoji'] = self[text_col].apply(get_emojis)

    def get_all_metrics(self,
                        text_col='text',
                        string=True,
                        words=True,
                        characters=True,
                        sentiment=False,
                        tags=False,
                        stopword_set=None,
                        senti_labels=None
                        ) -> None:
        '''
        Single function call to get and extract information from text.

        `stopword_set` : (set)
        default, `STOPWORDS=spacy.lang.en.STOP_WORDS`

            Used to capture metrics of actual words and stopwords
            detected within a string sequence.

        `senti_labels` : (tuple)
        default, `SENTI_LABELS=('positive', 'negative', 'neutral')`

            Used to label a text for every index in a column from a
            table containing sentiment values. Pass a tuple to change
            the default labels to use. e.g. (1, 0, 'N')

        String-level metric -> if string=True:
        --------------------------------------

        * `stringLength`  : sum of all words in a string.

        Word-level metrics -> if words=True:
        ------------------------------------

        * `avgWordLength`     : average number of words.
        * `isStopwordCount`   : count of stopwords only.
        * `noStopwordCount`   : count of none stopwords.

        Character-level metrics -> if character=True:
        -----------------------------------------------

        * `charDigitCount`    : count of digits chars.
        * `charUpperCount`    : count of uppercase chars.
        * `charLowerCount`    : count of lowercase chars.

        Sentiment-level metrics -> if sentiment=True:
        --------------------------------------------

        * `sentiPolarity`     : polarity score with Textblob, (float).
        * `sentiSubjectivity` : subjectivity score with Textblob (float).
        * `sentimentLabel`    : labels row with one (pos, neg, neutral) tag.

        Tag-extraction metrics -> if tags=True:
        ------------------------------------------

        * `authorTimeTag` : extracts video time tags, e.g. 1:20.
        * `authorUrlLink` : extracts urls links if found.
        * `authorHashTag` : extracts hash tags, e.g. #numberOne.
        * `authorEmoji`   : extracts emojis if found 👾.

        '''
        # overrides the default sets to used in multiple methods.
        if stopword_set:
            self.STOPWORDS = stopword_set
        if senti_labels:
            self.SENTI_LABELS = senti_labels

        self.strip_spaces(text_col)
        if string:
            self.string_metric(text_col)
        if words:
            self.word_metrics(text_col)
        if characters:
            self.character_metrics(text_col)
        if sentiment:
            self.sentiment_metrics(text_col)
        if tags:
            self.extract_authortags(text_col)

    def __repr__(self):
        return f'{ type(self) }'
