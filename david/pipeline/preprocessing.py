import re as _re
import string as _string
import contractions as _contractions
from itertools import groupby
from nltk.stem.wordnet import WordNetLemmatizer as _WordNetLemmatizer


def prep_textcolumn(df: object, text_col: str):
    '''NOTE: MOST IMPORTANT STEP BEFORE PREPROCESSING.
    MAKE SURE TO TEST TEXT SPACING! DO TESTS FOR THIS.
    WORK ON MAKING SURE ALL WHITE SPACING IS STRIPPED
    PROPERLY. EVEN WHEN DECIDING NOT TO REMOVE NONE-WORDS
    '''
    df[text_col] = df[text_col].str.strip()
    return df


def _trim_whitespaces(text):
    try:
        text = _re.sub(r"'", r"'", text)
        text = " ".join(text.split())
    except SyntaxError() as err:
        raise Exception(
            f"error, column with empty values in df: {err}")
        pass
    return text


def remove_excessive_spaces(text: str):
    """Remove more than one space"""
    # text =
    text = _re.sub(r'(\s)\1{1,}', ' ', text)
    text = text.strip()
    return text


def remove_duplicate_words(word: str):
    """Recommended use after preprocessing texts. For better results
    `Lemmatization` is one way to further improve word matches.
    Returns strings with no repeated words in sequence.

    Example ::
        >>> text = "Hey! you are wrong very wrong! wrong!"
        >>> text = remove_duplicate_words(text)
        >>> "Hey you are wrong very wrong"
    """
    # remove punctuation
    word_map = word.maketrans(dict.fromkeys(_string.punctuation))
    word_clean = word.translate(word_map)
    return ' '.join([k for k, v in groupby(word_clean.split())])


def reduce_repeatingchars(text: str):
    """Reduces repeated `characters`
    """
    findings = _re.findall(r'(\w)\1{2,}', text)
    for char in findings:
        find = char + '{3,}'
        replace = '???' + char + '???'
        text = _re.sub(find, repr(replace), text)
    text = text.replace('\'???', '')
    text = text.replace('???\'', '')
    text = remove_excessive_spaces(text)
    return text


def replace_contractions(df: object, text_col: str):
    """Replaces contractions (including slang words)
    NOTE: This process requires normal spacing between characters
    So trimming white spaces is applied before replacing contractions
    """
    df[text_col] = df[text_col].apply(lambda x: _trim_whitespaces(x))
    df[text_col] = df[text_col].apply(lambda x: _contractions.fix(x))
    return df


def standardize_text(df: object, text_col: str, lower_text: bool):
    df[text_col] = df[text_col].str.replace(r" '", r"'")
    df[text_col] = df[text_col].str.replace(r"http\S+", "")
    df[text_col] = df[text_col].str.replace(r"http", "")
    df[text_col] = df[text_col].str.replace(r"@\S+", "")
    df[text_col] = df[text_col].str.replace(
        r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    if lower_text:
        df[text_col] = df[text_col].str.lower()
    return df


def _TextLemmatizer(text: list):
    WordNet = _WordNetLemmatizer()
    return ' '.join([WordNet.lemmatize(word) for word in text])


def lemmetize_text(df: object, text_col: str):
    df[text_col] = df[text_col].str.split()
    df[text_col] = df[text_col].apply(lambda x: _TextLemmatizer(x))
    return df


def normalize_text(df: object, text_col: str):
    # df[text_col] = df[text_col].str.replace(r"[^a-zA-Z]", " ")
    df[text_col] = df[text_col].str.replace(
        r"&lt;/?.*?&gt;", " &lt;&gt; "
    )
    df[text_col] = df[text_col].str.replace(r"(\\d|\\W)+", " ")
    df[text_col] = df[text_col].str.replace(r"[^a-zA-Z]", " ")
    return df


def remove_duplicate_words_and_chars(df: object, text_col: str):
    df[text_col] = df[text_col].apply(
        lambda x: remove_duplicate_words(x))
    df[text_col] = df[text_col].apply(
        lambda x: reduce_repeatingchars(x))
    return df


def cleantext(
    df: object, text_col: str, lowertext=True,
    rm_contract=True, lemma=True, rm_duplicates=True
):
    """Dataframe Text Preprocessing Method. The arrangements
    have been crafted uniquely for Youtube Comments. Though,
    It can process similar styles from social-media text;
    (e.g. tweets or Reddit-posts). Cleaning text is recommended for
    increasing the accuracy/results from NLP models (e.g.
    `word2vec`, & `Gensim's LDA`).

    `NOTE`: The default parameters are recommended
    for good results.

    `DEV`: Make sure white spacing is a feature is done
    properly in many conditions!!

    PARAMETERS
    ----------
    `df` : (object, pandas.Dataframe)
        Pass a pandas dataframe must contain the text column!

    `text_col` : (str, df['dataframe column'])
        Pass the name of the column where the text is located.

    `lowertext` : (bool, default=True)
        Converts all texts to lowercase.

    `rm_contract` : (bool, default=True)
        Replaces common contractions (`including slang words`)

    `lemma` : (bool, default=True)
        Lemmatizes words: Useful for models like `LDA` and `word2vec`.

    `rm_duplicates` : (bool, default=True)
        Removes duplicate words (`words next-to-eachother ONLY`). It also
        reduces words with chars repeated more than 3 times to a single char.
        Useful to replace words such as `loooooooong` by `long`. `NOTE`: It can
        change abbreviations such as `AAA` to single `A`

            Example:
            >>> text = "Hey! you are wrong very wrong! wrong!"
            >>> text = remove_duplicate_words_and_chars(text)
            >>> "Hey you are wrong very wrong"

    ABOUT-NORMALIZER
    ----------------
    * Normalizes corpus by trimming whitespaces,
    removes all numbers urls, hashtags, mention
    tags found in the columns.
    """
    df = prep_textcolumn(df, text_col)
    if rm_contract:
        df = replace_contractions(df, text_col)
    df = standardize_text(df, text_col, lowertext)
    if lemma:
        df = lemmetize_text(df, text_col)
    df = normalize_text(df, text_col)
    if rm_duplicates:
        df = remove_duplicate_words_and_chars(df, text_col)
    return df
