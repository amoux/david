
import os
import re
import string
import unicodedata
from collections import Counter
from itertools import groupby
from typing import List, AnyStr

import contractions
import emoji
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from textblob import TextBlob

# inflect: correctly generate plurals, singular
# nouns, ordinals, indefinite articles


def strip_html_fromtext(text: str):
    soup = BeautifulSoup(text, "html.parser")
    soup = soup.get_text()
    # regex: removes html between brackets.
    text = re.sub(r'\[[^]]*\]', '', soup)
    return text


def get_emojis(str: str):
    '''Finds emoji characters from a sequence of words. Returns the emoji
    character if found.

    Usage:
    -----

        >>> text.get_emojis('this text has an emoji 👾.')
        '👾'
    '''
    emojis = ''.join(e for e in str if e in emoji.UNICODE_EMOJI)
    return emojis


def get_sentiment_polarity(text: str):
    return TextBlob(text).sentiment.polarity


def get_sentiment_subjectivity(text: str):
    return TextBlob(text).sentiment.subjectivity


def get_vocab_size(text: str):
    '''Returns the number of unique tokens found on the corpus.

    `text` : (str)
        The string that holds the entire corpus.
    '''
    word_map = Counter(text.split())
    unique_words = len(word_map.keys())
    vocab_size = int(unique_words)
    return vocab_size


def replace_numbers(words: List[str],
                    wantlist=False,
                    group=0,
                    comma=",",
                    andword="and",
                    zero="zero",
                    one="one",
                    decimal="point",
                    threshold=None) -> List[str]:
    '''
    Replace all interger occurrences in list of tokenized
    words with textual representation.

    * Returns a formatted number as it's word representation.

    Parameters:
    ----------

    `words` : (list)
        NOTE: The list must have tokenized words for the numbers
        to be replaced to words.

    `group` : (int)
        1, 2 or 3 to group numbers before turning into words.

    `comma` : (str)
        Define comma.

    `andword` : (str)
        Word for 'and' can be set to ''. e.g. "one hundred and
        one" vs "one hundred one".

    `zero` : (str)
        Word for '0'.

    `one` : (str)
        Word for '1'.

    `decimal` : (str)
        Word for decimal point.

    `threshold` :
        Numbers above threshold not turned into words.
        parameters not remembered from last call.

    Usage:
    -----

        >>> tokens = text.tokenizer('i would love it 4 sure!')
        >>> tokens
        '['i', 'would', 'love', 'it', '4', 'sure', '!']'

        >>> text.replace_numbers(tokens)
        '['i', 'would', 'love', 'it', 'four', 'sure', '!']'

    '''
    if not isinstance(words, List[str]):
        words = list([words])

    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word, wantlist,
                                         group, comma,
                                         andword, zero,
                                         one, decimal, threshold)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def replace_contractions(text: str, leftovers=True, slang=True):
    '''Replaces contractions (including slang words).

    NOTE: This process requires normal spacing between characters
    in order to properly replace words from a word sequence.

    '''
    return contractions.fix(text, leftovers=leftovers, slang=slang)


def normalize_spaces(text: str):
    '''Returns a string sequence with only one space.
    '''
    return ' '.join(t for t in text.split())


def remove_non_ascii(words: List[str]) -> List[str]:
    '''Remove non-ASCII characters from list of tokenized words.

    `words` : (list)
        List of tokenized words.
    '''
    if not isinstance(words, list):
        words = list([words])

    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: List[str]) -> List[str]:
    '''Remove punctuation from list of tokenized words.

    `words` : (list)
        The list can a list of tokens or a list of string(s).
    '''
    if not isinstance(words, list):
        words = list([words])

    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_duplicate_words(text: str):
    '''Returns strings with no repeated words in sequence.

    NOTE: This also removes punctuation.

    Example:
    -------
        >>> text = 'Hey! you are wrong very wrong! wrong!'
        >>> text = remove_duplicate_words(text)
        ...
        'Hey you are wrong very wrong'

    '''
    word_map = text.maketrans(dict.fromkeys(string.punctuation))
    word_clean = text.translate(word_map)
    return ' '.join([k for k, v in groupby(word_clean.split())])


def reduce_repeating_chars(text: str):
    '''Reduces repeated `characters`.
    '''
    findings = re.findall(r'(\w)\1{2,}', text)
    for char in findings:
        find = char + '{3,}'
        replace = '???' + char + '???'
        text = re.sub(find, repr(replace), text)
    text = text.replace('\'???', ' ')
    text = text.replace('???\'', ' ')
    text = normalize_spaces(text)
    return text


def remove_stopwords(words: List[str]) -> List[str]:
    '''Remove stop words from list of tokenized words
    '''
    if not isinstance(words, list):
        words = list([words])

    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stemmer(texts: List[str]) -> List[str]:
    Lancaster = LancasterStemmer()
    return ' '.join([Lancaster.stem(t) for t in texts])


def lemmatizer(texts: List[str]) -> List[str]:
    WordNet = WordNetLemmatizer()
    return ' '.join([WordNet.lemmatize(t) for t in texts])


def tokenizer(text: str) -> List[str]:
    '''
    Return the tokens of a sentence including punctuation:

        >>> tokenize('The apple. Where is the apple?')
    '['The', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']'
    '''
    return [t.strip() for t in re.split(r'(\W+)?', text) if t.strip()]

# NEW FUNCTIONS


# def replace_with_label(regex, label: str, text: str) -> AnyStr:
#     label = (f' {label} ')
#     regex = re.compile(regex)
#     text = re.sub(regex, label, text)
#     return text


# def replace_username_labels(text: str, label: str = None) -> AnyStr:
#     '''Replaces @usernames/handles from text by a given label.

#     >>> replace_handles('@GamerH Subscribed to your channel')
#         'USERNAME Subscribed to your channel'
#     '''
#     if not label:
#         label = (' USERNAME ')
#     text = re.sub(r'\B(\@[a-zA-Z_0-9]+\b)(?!;)', label, text)
#     text = re.sub(r'(\@[a-zA-Z0-9_%]*)', label, text)
#     text = normalize_spaces(text)
#     return text


# def remove_username_labels(text: str) -> AnyStr:
#     '''Removes @usernames/handles from text.'''
#     text = replace_username_labels(text=text, label='')
#     text = normalize_spaces(text=text)
#     return text


# def replace_quotes(text: str, label: str = None) -> AnyStr:
#     '''Replaces quotation marks with a label QUOTED.
#     '''
#     if not label:
#         label = (' QUOTED ')
#     text = re.sub(r'"', label, text)
#     text = re.sub(r"'", label, text)
#     return text


# def replace_contractions(text: str) -> AnyStr:
#     '''If two tokens are the same, this will merge them.

#     >>> text = "y'all thik he's python coding but ur o'l"
#     >>> re_contractions(text)
#     'you all thik he is python coding but you are old'
#     '''
#     text = (f' {text} ')
#     for k, v in EnglishContractions.items():
#         text = re.sub(re.escape(k), v, text, flags=re.I)
#     return text.strip()


# def spacy2textfile(fn, docs: list) -> None:
#     with open(fn, 'w', encoding='utf-8') as f:
#         for text in docs:
#             docs = nlp(text)
#             for sent in docs.sents:
#                 if len(sent) > 1:
#                     f.write('%s\n' % sent)
#         f.close()


# def iter_text2sents(text: str):
#     sentences = []
#     doc = nlp(text)
#     for sent in doc.sents:
#         if len(sent) > 1:
#             sentences.append(sent)
#     return sentences
