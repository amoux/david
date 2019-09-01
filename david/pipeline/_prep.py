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

    def remove_spaces(self, text: str):
        '''Remove more than one space.
        '''
        return re.sub(r'(\s)\1{1,}', ' ', text).strip()

    def remove_duplicate_words(self, text: str):
        '''Returns strings with no repeated words in sequence.
        NOTE: This also removes punctuation.

        Example:
        -------
            >>> text = 'Hey! you are wrong very wrong! wrong!'
            >>> text = remove_duplicate_words(text)
            ...
            'Hey you are wrong very wrong'
        '''
        # removes punctuation
        word_map = text.maketrans(dict.fromkeys(string.punctuation))
        word_clean = text.translate(word_map)
        return ' '.join([k for k, v in groupby(word_clean.split())])

    def reduce_repeating_chars(self, text: str):
        '''Reduces repeated `characters`.
        '''
        findings = re.findall(r'(\w)\1{2,}', text)
        for char in findings:
            find = char + '{3,}'
            replace = '???' + char + '???'
            text = re.sub(find, repr(replace), text)
        text = text.replace('\'???', '')
        text = text.replace('???\'', '')
        text = self.remove_spaces(text)
        return text

    def tokenizer(self, text: str) -> List[str]:
        '''Return the tokens of a sentence including punctuation:

            >>> tokenize('The apple. Where is the apple?')
        '['The', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']'
        '''
        return [x.strip() for x in re.split(r'(\W+)?', text) if x.strip()]

    def lemmatizer(self, texts: list):
        WordNet = WordNetLemmatizer()
        return ' '.join([WordNet.lemmatize(w) for w in texts])

    def lower_texts(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.lower()

    def replace_contractions(self, text_col='text'):
        '''Replaces contractions (including slang words)

        NOTE: This process requires normal spacing between characters.
        '''
        self[text_col] = self[text_col].apply(lambda x: contractions.fix(x))

    def normalize_texts(self, text_col='text') -> None:
        self[text_col] = self[text_col].apply(
            lambda x: self.remove_duplicate_words(x))
        self[text_col] = self[text_col].apply(
            lambda x: self.reduce_repeating_chars(x))

    def _standardize_text_A(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.replace(r" '", r"'")
        self[text_col] = self[text_col].str.replace(r"http\S+", "")
        self[text_col] = self[text_col].str.replace(r"http", "")
        self[text_col] = self[text_col].str.replace(r"@\S+", "")
        self[text_col] = self[text_col].str.replace(
            r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")

    def _standardize_text_B(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.replace(
            r"&lt;/?.*?&gt;", " &lt;&gt; ")
        self[text_col] = self[text_col].str.replace(r"(\\d|\\W)+", " ")
        self[text_col] = self[text_col].str.replace(r"[^a-zA-Z]", " ")

    def lemmetize_texts(self, text_col='text') -> None:
        self[text_col] = self[text_col].str.split()
        self[text_col] = self[text_col].apply(
            lambda x: self.lemmatizer(x))

    def tokenize_texts(self, text_col='text') -> None:
        self[text_col] = self[text_col].apply(lambda x: self.tokenizer(x))

    def clean_all_text(self,
                       text_col='text',
                       standardize=True,
                       contractions=True,
                       lemmatize=False,
                       normalize=True,
                       lower_texts=False,
                       tokenize=False
                       ) -> None:
        '''
        Dataframe Text Preprocessing Method. The arrangements
        have been crafted uniquely for Youtube Comments. Though,
        It can process similar styles from social-media text;
        (e.g. tweets or Reddit-posts). Cleaning text is recommended for
        increasing the accuracy/results from NLP models (e.g.
        `word2vec`, & `Gensim's LDA`).

        Parameters:
        ----------

        `text_col` : (str) default='text'
            Pass the name of the column where the text is located.

        `standardize` : (bool)
            Standardizes by trimming whitespaces, numbers, urls, hashtags,
            mention and tags found in the text.

        `contractions` : (bool, default=True)
            Replaces common contractions (`including slang words`)

        `lemmatize` : (bool, default=True)
            Lemmatizes words: Useful for models like `LDA` and `word2vec`.

        `normalize` : (bool, default=True)
            Removes duplicate words (`words next-to-eachother ONLY`).
            It also reduces words with chars repeated more than 3 times
            to a single char. Useful to replace words such as `looooooong`
            by `long`. `NOTE`: It can change abbreviations such as `AAA`
            to single `A`.

        `lower_texts` : (bool, default=True)
            Converts all texts to lowercase.

        `tokenize` : (bool) default=False
            Converts a string into a list or array like.
        '''
        self.normalize_whitespaces(text_col)
        if contractions:
            self.replace_contractions(text_col)

        if standardize:

            # NOTE: improve these methods names
            # and the order and instention to what
            # they use each regex pattern.

            self._standardize_text_A(text_col)
            self._standardize_text_B(text_col)

        if lemmatize:
            self.lemmetize_texts(text_col)
        if normalize:
            self.normalize_texts(text_col)
        if lower_texts:
            self.lower_texts(text_col)
        if tokenize:
            self.tokenize_texts(text_col)
