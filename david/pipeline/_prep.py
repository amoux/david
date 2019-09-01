import re
import string
from itertools import groupby
from typing import List

import contractions
from nltk.stem.wordnet import WordNetLemmatizer

from .base import JsonDataFrame


class TextPreprocess(JsonDataFrame):

    def __init__(self, corpus_path: str):
        super().__init__(corpus_path)

    def remove_whitespaces(self, text: str):
        '''Remove more than one space.
        '''
        return re.sub(r'(\s)\1{1,}', ' ', text).strip()

    def tokenize(self, text: str) -> List[str]:
        '''Return the tokens of a sentence including punctuation:

            >>> tokenize('The apple. Where is the apple?')
        '['The', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']'
        '''
        return [x.strip() for x in re.split(r'(\W+)?', text) if x.strip()]

    def remove_duplicatewords(self, text: str):
        '''Returns strings with no repeated words in sequence.

        Example:
        -------
            >>> text = 'Hey! you are wrong very wrong! wrong!'
            >>> text = remove_duplicatewords(text)
            ...
            'Hey you are wrong very wrong'
        '''
        # removes punctuation
        word_map = text.maketrans(dict.fromkeys(string.punctuation))
        word_clean = text.translate(word_map)
        return ' '.join([k for k, v in groupby(word_clean.split())])

    def reduce_repeatingchars(self, text: str):
        '''Reduces repeated `characters`.
        '''
        findings = re.findall(r'(\w)\1{2,}', text)
        for char in findings:
            find = char + '{3,}'
            replace = '???' + char + '???'
            text = re.sub(find, repr(replace), text)
        text = text.replace('\'???', '')
        text = text.replace('???\'', '')
        text = self.remove_whitespaces(text)
        return text

    def replace_contractions(self, text_col='text'):
        '''Replaces contractions (including slang words)
        NOTE: This process requires normal spacing between characters.
        '''
        self[text_col] = self[text_col].apply(lambda x: contractions.fix(x))

    def lemmatizer(self, texts: list):
        WordNet = WordNetLemmatizer()
        return ' '.join([WordNet.lemmatize(w) for w in texts])

    def remove_duplicatewords_andchars(self, text_col='text') -> None:
        self[text_col] = self[text_col].apply(
            lambda x: self.remove_duplicatewords(x))
        self[text_col] = self[text_col].apply(
            lambda x: self.reduce_repeatingchars(x))

    def lower_text(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.lower()

    def standardize_text(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.replace(r" '", r"'")
        self[text_col] = self[text_col].str.replace(r"http\S+", "")
        self[text_col] = self[text_col].str.replace(r"http", "")
        self[text_col] = self[text_col].str.replace(r"@\S+", "")
        self[text_col] = self[text_col].str.replace(
            r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")

    def lemmetize_texts(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.split()
        self[text_col] = self[text_col].apply(
            lambda x: self.lemmatizer(x))

    def normalize_texts(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.replace(
            r"&lt;/?.*?&gt;", " &lt;&gt; ")
        self[text_col] = self[text_col].str.replace(r"(\\d|\\W)+", " ")
        self[text_col] = self[text_col].str.replace(r"[^a-zA-Z]", " ")

    def clean_all_text(self, text_col='text', standardize=True,
                       contractions=True, lemmatize=False, normalize=True,
                       rm_duplicates=True, lower_text=False) -> None:
        '''
        Dataframe Text Preprocessing Method. The arrangements
        have been crafted uniquely for Youtube Comments. Though,
        It can process similar styles from social-media text;
        (e.g. tweets or Reddit-posts). Cleaning text is recommended for
        increasing the accuracy/results from NLP models (e.g.
        `word2vec`, & `Gensim's LDA`).

        Parameters:
        ----------

        `text_col` : (str, df['dataframe column'])
            Pass the name of the column where the text is located.

        `lowertext` : (bool, default=True)
            Converts all texts to lowercase.

        `contractions` : (bool, default=True)
            Replaces common contractions (`including slang words`)

        `lemmatize` : (bool, default=True)
            Lemmatizes words: Useful for models like `LDA` and `word2vec`.

        `rm_duplicates` : (bool, default=True)
            Removes duplicate words (`words next-to-eachother ONLY`).
            It also reduces words with chars repeated more than 3 times
            to a single char. Useful to replace words such as `looooooong`
            by `long`. `NOTE`: It can change abbreviations such as `AAA`
            to single `A`.

        About Normalizer:
        ----------------

            Normalizes corpus by trimming whitespaces,
            removes all numbers urls, hashtags, mention
            tags found in the columns.

        '''
        self.normalize_whitespaces(text_col)
        if contractions:
            self.replace_contractions(text_col)
        if standardize:
            self.standardize_text(text_col)
        if lemmatize:
            self.lemmetize_texts(text_col)
        if normalize:
            self.normalize_texts(text_col)
        if rm_duplicates:
            self.remove_duplicatewords_andchars(text_col)
        if lower_text:
            self.lower_text(text_col)
