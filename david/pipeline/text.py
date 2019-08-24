
import os
import re
import unicodedata

import contractions
import inflect
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# inflect: correctly generate plurals, singular
# nouns, ordinals, indefinite articles


def _html_textparser(text: str):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def _html_betweenbrackets(text: str):
    return re.sub(r'\[[^]]*\]', '', text)


def strip_htmlfromtext(text: str):
    text = _html_textparser(text)
    text = _html_betweenbrackets(text)
    return text


def replace_contractions(text: str):
    '''Replace contractions in string of text
    '''
    return contractions.fix(text)


def remove_non_ascii(words: list):
    '''Remove non-ASCII characters from list of tokenized words
    '''
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words


def to_lowercase(words: list):
    '''
    Convert all characters to lowercase from
    list of tokenized words
    '''
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


def remove_punctuation(words: list):
    '''Remove punctuation from list of tokenized words
    '''
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words


def replace_numbers(words: list):
    '''
    Replace all interger occurrences in list of
    tokenized words with textual representation
    '''
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words


def remove_stopwords(words: list):
    '''Remove stop words from list of tokenized words
    '''
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words: list):
    '''Stem words in list of tokenized words
    '''
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


def lemmatize_verbs(words: list):
    '''Lemmatize verbs in list of tokenized words
    '''
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def normalize(words: list):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


def stem_and_lemmatize(words: list):
    '''
    RETURNS:
    stems, lemmas = stem_and_lemmatize(words)
    '''
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems, lemmas
