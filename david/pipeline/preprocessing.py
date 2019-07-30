import re as _re
import string as _string
import contractions as _contractions
from itertools import groupby
from nltk.stem.wordnet import WordNetLemmatizer as _WordNetLemmatizer


def prep_textcolumn(df: object, text_col: str):
    '''Prep texts by normalizing whitespaces.
    '''
    df[text_col] = df[text_col].str.strip()
    return df


def remove_whitespaces(text: str):
    '''Remove more than one space.
    '''
    text = _re.sub(r'(\s)\1{1,}', ' ', text)
    text = text.strip()
    return text


def remove_duplicatewords(word: str):
    '''Recommended use after preprocessing texts. For better results
    `Lemmatization` is one way to further improve word matches.
    Returns strings with no repeated words in sequence.

    Example ::
        >>> text = "Hey! you are wrong very wrong! wrong!"
        >>> text = remove_duplicatewords(text)
        >>> "Hey you are wrong very wrong"
    '''
    # remove punctuation
    word_map = word.maketrans(dict.fromkeys(_string.punctuation))
    word_clean = word.translate(word_map)
    return ' '.join([k for k, v in groupby(word_clean.split())])


def reduce_repeatingchars(text: str):
    '''Reduces repeated `characters`
    '''
    findings = _re.findall(r'(\w)\1{2,}', text)
    for char in findings:
        find = char + '{3,}'
        replace = '???' + char + '???'
        text = _re.sub(find, repr(replace), text)
    text = text.replace('\'???', '')
    text = text.replace('???\'', '')
    text = remove_whitespaces(text)
    return text


def replace_contractions(df: object, text_col: str):
    '''Replaces contractions (including slang words)
    NOTE: This process requires normal spacing between characters.
    '''
    df[text_col] = df[text_col].apply(lambda x: _contractions.fix(x))
    return df


def _TextLemmatizer(text: list):
    WordNet = _WordNetLemmatizer()
    return ' '.join([WordNet.lemmatize(w) for w in text])


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


def lemmetize_texts(df: object, text_col: str):
    df[text_col] = df[text_col].str.split()
    df[text_col] = df[text_col].apply(lambda x: _TextLemmatizer(x))
    return df


def normalize_texts(df: object, text_col: str):
    df[text_col] = df[text_col].str.replace(
        r"&lt;/?.*?&gt;", " &lt;&gt; "
    )
    df[text_col] = df[text_col].str.replace(r"(\\d|\\W)+", " ")
    df[text_col] = df[text_col].str.replace(r"[^a-zA-Z]", " ")
    return df


def remove_duplicatewords_andchars(df: object, text_col: str):
    df[text_col] = df[text_col].apply(
        lambda x: remove_duplicatewords(x))
    df[text_col] = df[text_col].apply(
        lambda x: reduce_repeatingchars(x))
    return df


def cleantext(
    df: object, text_col: str, lowertext=True,
    rm_contract=True, lemma=False, rm_duplicates=True
):
    '''Dataframe Text Preprocessing Method. The arrangements
    have been crafted uniquely for Youtube Comments. Though,
    It can process similar styles from social-media text;
    (e.g. tweets or Reddit-posts). Cleaning text is recommended for
    increasing the accuracy/results from NLP models (e.g.
    `word2vec`, & `Gensim's LDA`).

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
            >>> text = remove_duplicatewords_andchars(text)
            >>> "Hey you are wrong very wrong"

    ABOUT-NORMALIZER
    ----------------
    * Normalizes corpus by trimming whitespaces,
    removes all numbers urls, hashtags, mention
    tags found in the columns.
    '''
    df = prep_textcolumn(df, text_col)
    if rm_contract:
        df = replace_contractions(df, text_col)
    df = standardize_text(df, text_col, lowertext)
    if lemma:
        df = lemmetize_texts(df, text_col)
    df = normalize_texts(df, text_col)
    if rm_duplicates:
        df = remove_duplicatewords_andchars(df, text_col)
    return df
