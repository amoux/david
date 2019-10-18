
from collections import MutableSequence

import numpy as np
from spacy.lang.en import STOP_WORDS as SPACY_STOPWORDS

from ..text import (get_emojis, get_sentiment_polarity,
                    get_sentiment_subjectivity, normalize_spaces)

# Regex patterns used. (replacing these soon)
TIME = r'(\d{1,2}\:\d{1,2})'
URL = r'(http\S+)'
TAG = r'(\#\w+)'


class TextMetrics(MutableSequence, object):
    """Gathers Statistical Metrics from Texts."""

    STOPWORDS = SPACY_STOPWORDS
    SENTI_LABELS = ('positive', 'negative', 'neutral')

    def sentiment_labeler(self, score):
        if (score > 0):
            return self.SENTI_LABELS[0]
        if (score < 0):
            return self.SENTI_LABELS[1]
        else:
            return self.SENTI_LABELS[2]

    def avg_words(self, words: list):
        return [len(w) for w in words.split(' ') if w not in self.STOPWORDS]

    def string_metric(self, text_col='text'):
        self[text_col] = self[text_col].apply(lambda s: normalize_spaces(s))
        self['stringLength'] = self[text_col].str.len()

    def word_metrics(self, text_col='text'):

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
        self['sentiPolarity'] = self[text_col].apply(
            get_sentiment_polarity)
        self['sentiSubjectivity'] = self[text_col].apply(
            get_sentiment_subjectivity)
        self['sentimentLabel'] = self['sentiPolarity'].apply(
            lambda x: self.sentiment_labeler(x))

    def extract_authortags(self, text_col='text'):
        self['authorTimeTag'] = self[text_col].str.extract(TIME)
        self['authorUrlLink'] = self[text_col].str.extract(URL)
        self['authorHashTag'] = self[text_col].str.extract(TAG)
        self['authorEmoji'] = self[text_col].apply(get_emojis)

    def get_all_metrics(
            self,
            text_col='text',
            string=True,
            words=True,
            characters=True,
            sentiment=False,
            tags=False,
            stopword_set=None,
            senti_labels=None) -> None:
        """Single function call to extract information from text.

        Parameters:
        -----------
        stopword_set : (set)
            default, STOPWORDS=spacy.lang.en.STOP_WORDS
            Used to capture metrics of actual words and stopwords
            detected within a string sequence.
        senti_labels : (tuple)
            default, SENTI_LABELS=('positive', 'negative', 'neutral')
            Used to label a text for every index in a column from a
            table containing sentiment values. Pass a tuple to change
            the default labels to use. e.g. (1, 0, 'N')

        String-level metric -> if string=True:
        --------------------------------------

        * stringLength  : sum of all words in a string.

        Word-level metrics -> if words=True:
        ------------------------------------

        * avgWordLength     : average number of words.
        * isStopwordCount   : count of stopwords only.
        * noStopwordCount   : count of none stopwords.

        Character-level metrics -> if character=True:
        ---------------------------------------------

        * charDigitCount    : count of digits chars.
        * charUpperCount    : count of uppercase chars.
        * charLowerCount    : count of lowercase chars.

        Sentiment-level metrics -> if sentiment=True:
        --------------------------------------------

        * sentiPolarity     : polarity score with Textblob, (float).
        * sentiSubjectivity : subjectivity score with Textblob (float).
        * sentimentLabel    : labels row with one (pos, neg, neutral) tag.

        Tag-extraction metrics -> if tags=True:
        --------------------------------------

        * authorTimeTag : extracts video time tags, e.g. 1:20.
        * authorUrlLink : extracts urls links if found.
        * authorHashTag : extracts hash tags, e.g. #numberOne.
        * authorEmoji   : extracts emojis if found 👾.

        """
        if stopword_set:
            self.STOPWORDS = stopword_set
        if senti_labels:
            self.SENTI_LABELS = senti_labels
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
