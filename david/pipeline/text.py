
import os
import re
import string
import unicodedata
from collections import Counter
from itertools import groupby
from typing import List

import contractions
import emoji
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from textblob import TextBlob

# inflect: correctly generate plurals, singular
# nouns, ordinals, indefinite articles


def to_lowercase(words: list):
    '''Convert all characters to lowercase from
    list of tokenized words.
    '''
    if not isinstance(words, list):
        words = list([words])

    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def strip_html_fromtext(text: str):
    soup = BeautifulSoup(text, "html.parser")
    soup = soup.get_text()
    # regex: removes html between brackets.
    text = re.sub(r'\[[^]]*\]', '', soup)
    return text


def get_emojis(str):
    '''Finds emoji characters from a sequence of words. Returns the emoji
    character if found.
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


def replace_numbers(words: list):
    '''Replace all interger occurrences in list of
    tokenized words with textual representation.

    Usage:
    -----

        >>> tokens = text.tokenizer('i would love it 4 sure!')
        >>> tokens
        '['i', 'would', 'love', 'it', '4', 'sure', '!']'

        >>> text.replace_numbers(tokens)
        '['i', 'would', 'love', 'it', 'four', 'sure', '!']'

    '''
    if not isinstance(words, list):
        words = list([words])

    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
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


def remove_spaces(text: str):
    '''Remove more than one space.
    '''
    return re.sub(r'(\s)\1{1,}', ' ', text).strip()


def remove_non_ascii(words: list):
    '''Remove non-ASCII characters from list of tokenized words.
    '''
    if not isinstance(words, list):
        words = list([words])

    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: list):
    '''Remove punctuation from list of tokenized words.
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
    text = text.replace('\'???', '')
    text = text.replace('???\'', '')
    text = remove_spaces(text)
    return text


def remove_stopwords(words: list):
    '''Remove stop words from list of tokenized words
    '''
    if not isinstance(words, list):
        words = list([words])

    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stemmer(texts: list):
    Lancaster = LancasterStemmer()
    return ' '.join([Lancaster.stem(w) for w in texts])


def lemmatizer(texts: list):
    WordNet = WordNetLemmatizer()
    return ' '.join([WordNet.lemmatize(w) for w in texts])


def tokenizer(text: str) -> List[str]:
    '''
    Return the tokens of a sentence including punctuation:

        >>> tokenize('The apple. Where is the apple?')
    '['The', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']'
    '''
    return [x.strip() for x in re.split(r'(\W+)?', text) if x.strip()]
