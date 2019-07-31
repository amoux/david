
import emoji as _emoji
import numpy as _np
from spacy.lang import en as _en
from textblob import TextBlob as _TextBlob

from .preprocessing import prep_textcolumn as _preptext

_STOPWORDS = _en.STOP_WORDS


def extract_emojis(str):
    emojis = ''.join(e for e in str if e in _emoji.UNICODE_EMOJI)
    return emojis


def sentiment_polarity(text: str):
    return _TextBlob(text).sentiment.polarity


def sentiment_subjectivity(text: str):
    return _TextBlob(text).sentiment.subjectivity


def sentiment_labeler(df: object):
    """Assigns sentiment label to each text-row
    from the Textblob scores.
    """
    labels = []
    for score in df['sentiPolarity']:
        if score > 0:
            labels.append("positive")
        if score < 0:
            labels.append("negative")
        elif score == 0:
            labels.append("neutral")
    return labels


def count_none_stopwords(df: object, text_col: str):
    '''Returns the count of words not consider as `STOPWORDS`.

    Uses the set found in Spacy's API; `spacy.lang.en.STOPWORDS`.
    '''
    df['notStopwordsCount'] = df[text_col].apply(lambda texts: len(
        [w for w in texts.split(' ') if w not in _STOPWORDS]))
    return df


def avgword_length(df: object, text_col: str):
    '''Returns the average word-length in a text column.
    NOTE: The returned value is without words consider as `STOPWORDS`.

    Uses the set found in Spacy's API; `spacy.lang.en.STOPWORDS`.
    '''
    df['wordAvgLength'] = df[text_col].apply(lambda texts: _np.mean(
        [len(w) for w in texts.split(' ') if w not in _STOPWORDS]
    ) if len([len(w) for w in texts.split(' ') if w not in _STOPWORDS]
             ) > 0 else 0)
    return df


def extract_textmetrics(df: object, text_col: str):
    # wordCounts: -1 to get an exact value of word counts in the string
    df['hasStopwordsCount'] = df[text_col].apply(lambda x: len(str(x).split()))
    df['wordStrLen'] = df[text_col].str.len()
    df['charIsDigitCount'] = df[text_col].str.findall(r'[0-9]').str.len()
    df['charIsUpperCount'] = df[text_col].str.findall(r'[A-Z]').str.len()
    df['charIsLowerCount'] = df[text_col].str.findall(r'[a-z]').str.len()
    df = count_none_stopwords(df, text_col)
    df = avgword_length(df, text_col)
    return df


def extract_authortags(df: object, text_col: str):
    df['authorTimeTag'] = df[text_col].str.extract(r'(\d{1,2}\:\d{1,2})')
    df['authorUrlLink'] = df[text_col].str.extract(r'(http\S+)')
    df['authorHashTag'] = df[text_col].str.extract(r'(\#\w+)')
    df['authorEmoji'] = df[text_col].apply(extract_emojis)
    return df


def sentiment_fromtexts(df: object, text_col: str):
    df['sentiPolarity'] = df[text_col].apply(sentiment_polarity)
    df['sentiSubjectivity'] = df[text_col].apply(sentiment_subjectivity)
    df['sentimentLabel'] = sentiment_labeler(df)
    return df


def reduce_dataframesize(df: object, by_wordcount: int):
    # NOTE:test function with an random empty values.
    # dataframe = dataframe[dataframe['word_count'] > int(by_wordcount)]
    df[df['word_count'] > int(by_wordcount)]
    return df


def textmetrics(df: object, text_col: str, gettags=False, sentiment=False,
                min_wordcount: int = 0
                ):
    """Gathers Statistical Metrics from Texts.
    The collection of functions only work by passing a dataframe object.
    Returns a data frame with new rows holding various text metrics.

    PARAMETERS
    ----------

    `df` : (object)
        Pass a pandas.Dataframe object holding a text-column

    `text_col` : (list)
        The name of the column in the pandas.Dataframe containing
        texts.

    `min_wordcount` : (int)
        The minimum number of characters/words counts to use as a way to
        reduce/slice the size of the data frame. Valuable sentences hold
        at slightest a `wordCount` greater than `10`.

    METRICS
    -------

        >>> `wordCount` : 'counts number of words in the text.'
        >>> `wordStrLen` : 'the sum of words (length) in the text.'
        >>> `charIsDigitCount` : 'count of digits chars.'
        >>> `charIsUpperCount` : 'count of uppercase chars.'
        >>> `charIsLowerCount` : 'count of lowercase chars.'

    * (additional-metrics) if `gettags` = `True`:

        >>> `authorTimeTag` : 'extracts video time tags, e.g. 1:20.'
        >>> `authorUrlLink` : 'extracts urls links if found.'
        >>> `authorHashTag` : 'extracts hash tags, e.g. #numberOne
        >>> `authorEmoji`   : 'extracts emojis if found.'

    * (additional-metrics) if `sentiment` = `True`:

        >>> `sentiPolarity`     : 'polarity score with Textblob, (float).'
        >>> `sentiSubjectivity` : 'subjectivity score with Textblob, (float).'
        >>> `sentimentLabel`    : 'labels row w\one (pos, neg, neutral) tag.'
    """
    df = _preptext(df, text_col)
    df = extract_textmetrics(df, text_col)
    if gettags:
        df = extract_authortags(df, text_col)
    if sentiment:
        df = sentiment_fromtexts(df, text_col)
    if min_wordcount > 0:
        df = reduce_dataframesize(df, min_wordcount)
    return df
