
import emoji as _emoji
from textblob import TextBlob as _TextBlob
from .preprocessing import prep_textcolumn as _preptext


def trim_whitespaces(text):
    try:
        text = " ".join(text.split())
    except ZeroDivisionError:
        pass
    return text


def avgwords(words):
    # this function is not necessary... removing it for now.
    try:
        avgwords = (sum(len(w.split()) for w in words)/len(words))
    except ZeroDivisionError as err:
        msg = f"Division error, column found with 0 values: {err}"
        raise Exception(msg)
    else:
        return avgwords


def extract_emojis(str):
    emojis = ''.join(x for x in str if x in _emoji.UNICODE_EMOJI)
    return emojis


def get_polarityscore(text):
    polarity = _TextBlob(text).sentiment.polarity
    return polarity


def get_subjectivityscore(text):
    subjectivity = _TextBlob(text).sentiment.subjectivity
    return subjectivity


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


def extract_textmetrics(df: object, text_col: str):
    df['text'] = df[text_col].apply(trim_whitespaces)
    # wordCounts: [-1] to get an exact value of word counts in the string
    df['wordCount'] = df[text_col].apply(lambda x: (len(str(x).split()))-1)
    df['wordStrLen'] = df[text_col].str.len()
    df['charIsDigitCount'] = df[text_col].str.findall(r'[0-9]').str.len()
    df['charIsUpperCount'] = df[text_col].str.findall(r'[A-Z]').str.len()
    df['charIsLowerCount'] = df[text_col].str.findall(r'[a-z]').str.len()
    return df


def extract_authortags(df: object, text_col: str):
    df['authorTimeTag'] = df[text_col].str.extract(r'(\d{1,2}\:\d{1,2})')
    df['authorUrlLink'] = df[text_col].str.extract(r'(http\S+)')
    df['authorHashTag'] = df[text_col].str.extract(r'(\#\w+)')
    df['authorEmoji'] = df[text_col].apply(extract_emojis)
    return df


def sentiment_fromtexts(df: object, text_col: str):
    df['sentiPolarity'] = df[text_col].apply(get_polarityscore)
    df['sentiSubjectivity'] = df[text_col].apply(get_subjectivityscore)
    df['sentimentLabel'] = sentiment_labeler(df)
    return df


def reduce_dataframesize(df: object, by_wordcount: int):
    # NOTE:test function with an random empty values.
    # dataframe = dataframe[dataframe['word_count'] > int(by_wordcount)]
    df[df['word_count'] > int(by_wordcount)]
    return df


def textmetrics(df: object, text_col: str, gettags=False, sentiment=False,
                min_wordcount=0
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
