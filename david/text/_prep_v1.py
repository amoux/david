
import os
import re
import string
import unicodedata
from collections import Counter
from itertools import groupby
from typing import AnyStr, List

import contractions
import emoji
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from textblob import TextBlob


def get_emojis(str: str):
    emojis = ''.join(e for e in str if e in emoji.UNICODE_EMOJI)
    return emojis


def get_sentiment_polarity(text: str):
    # NOTE: THESE ARE USELESS I WILL REMOVE THEM
    # BUT FIRST I NEED TO FIX THE METHODS USING IT.
    return TextBlob(text).sentiment.polarity


def get_sentiment_subjectivity(text: str):
    # NOTE: THESE ARE USELESS I WILL REMOVE THEM
    # BUT FIRST I NEED TO FIX THE METHODS USING IT.
    return TextBlob(text).sentiment.subjectivity


def get_vocab_size(text: str):
    word_map = Counter(text.split())
    unique_words = len(word_map.keys())
    vocab_size = int(unique_words)
    return vocab_size


def normalize_spaces(text: str):
    return ' '.join(t for t in text.split())


def remove_punctuation(words: list):
    # NOTE: THESE ARE USELESS I WILL REMOVE THEM
    # BUT FIRST I NEED TO FIX THE METHODS USING IT.
    if not isinstance(words, list):
        words = list([words])
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def remove_duplicate_words(text: str):
    """NOTE: REMOVING THIS FUNCTION NO LONGER NEEDED DUE TO
    NEW VERSION OF `_.prep_v2.reduce_repeating_chars`
    """
    word_map = text.maketrans(dict.fromkeys(string.punctuation))
    word_clean = text.translate(word_map)
    return ' '.join([k for k, v in groupby(word_clean.split())])


def reduce_repeating_chars_v1(text: str):
    """NOTE: REMOVING THIS FUNCTION AND REPLACING IT WITH
    THIS AN OPTIMIZED VERSION `_.prep_v2.reduce_repeating_chars`
    """
    findings = re.findall(r'(\w)\1{2,}', text)
    for char in findings:
        find = char + '{3,}'
        replace = '???' + char + '???'
        text = re.sub(find, repr(replace), text)
    text = text.replace('\'???', ' ')
    text = text.replace('???\'', ' ')
    text = normalize_spaces(text)
    return text


def stemmer(texts: list):
    Lancaster = LancasterStemmer()
    return ' '.join([Lancaster.stem(t) for t in texts])


def lemmatizer(texts: list):
    WordNet = WordNetLemmatizer()
    return ' '.join([WordNet.lemmatize(t) for t in texts])


def regex_tokenizer(text: str):
    return [t.strip() for t in re.split(r'(\W+)?', text) if t.strip()]
