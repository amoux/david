
from collections import MutableSequence

from ..text import (expand_contractions, lemmatizer, normalize_spaces,
                    reduce_repeating_chars_v1, regex_tokenizer,
                    remove_duplicate_words)


class TextPreprocess(MutableSequence, object):

    def strip_spaces(self, text_col: str = 'text'):
        self[text_col] = self[text_col].apply(
            lambda s: normalize_spaces(s))

    def lower_texts(self, text_col: str = 'text') -> None:
        self[text_col] = self[text_col].str.lower()

    def fix_contractions(self, text_col: str = 'text',
                         leftovers=True, slang=True):
        self.strip_spaces(text_col)
        self[text_col] = self[text_col].apply(
            lambda s: expand_contractions(s, leftovers, slang))

    def normalize_texts(self, text_col: str = 'text') -> None:
        self[text_col] = self[text_col].apply(
            lambda s: remove_duplicate_words(s))
        self[text_col] = self[text_col].apply(
            lambda s: reduce_repeating_chars_v1(s))

    def standardizerA(self, text_col: str = 'text') -> None:
        self[text_col] = self[text_col].str.replace(r"http\S+", "")
        self[text_col] = self[text_col].str.replace(r"@\S+", "")
        self[text_col] = self[text_col].str.replace(
            r"([^A-Za-z0-9(),!?@\'\`\"\_\n])", " ")

    def standardizerB(self, text_col: str = 'text') -> None:
        self[text_col] = self[text_col].str.replace(
            r"&lt;/?.*?&gt;", " &lt;&gt; ")
        self[text_col] = self[text_col].str.replace(r"(\\d|\\W)+", " ")
        self[text_col] = self[text_col].str.replace(r"[^a-zA-Z]", " ")

    def lemmetize_texts(self, text_col: str = 'text') -> None:
        self[text_col] = self[text_col].str.split()
        self[text_col] = self[text_col].apply(lambda s: lemmatizer(s))

    def tokenize_texts(self, text_col: str = 'text') -> None:
        self[text_col] = self[text_col].apply(lambda s: regex_tokenizer(s))

    def clean_all_text(
            self,
            text_col: str = 'text',
            standardize=True,
            contractions=True,
            leftovers=True,
            slang=True,
            lemmatize=False,
            normalize=True,
            lower_texts=False,
            tokenize=False) -> None:
        """Dataframe Text Preprocessing Method.

        The arrangements have been crafted uniquely for Youtube Comments.
        Though, It can process similar styles from social-media text;
        (e.g. tweets or Reddit-posts). Cleaning text is recommended for
        increasing the accuracy/results from NLP models (e.g.
        `word2vec`, & `Gensim's LDA`).

        Parameters:
        ----------
        text_col : (str) default='text'
            Pass the name of the column where the text is located.
        standardize : (bool)
            Standardizes by trimming whitespaces, numbers, urls, hashtags,
            mention and tags found in the text.
        contractions : (bool, default=True)
            Replaces common contractions (including slang words)
        slang : (bool, default=True)
            If False, slang words will be ignored and not replaced.
        lemmatize : (bool, default=True)
            Lemmatizes words: Useful for models like LDA and word2vec.
        normalize : (bool, default=True)
            Removes duplicate words (words next-to-eachother ONLY).
            It also reduces words with chars repeated more than 3 times
            to a single char.
        lower_texts : (bool, default=True)
            Converts all texts to lowercase.
        tokenize : (bool) default=False
            Converts a string into a list or array like.
        """
        self.strip_spaces(text_col)
        if contractions:
            self.fix_contractions(text_col, leftovers, slang)
        if standardize:
            # NOTE: improve these methods names
            # and the order and instention to what
            # they use each regex pattern.
            self.standardizerA(text_col)
            self.standardizerB(text_col)
        if normalize:
            self.normalize_texts(text_col)
        if lemmatize:
            self.lemmetize_texts(text_col)
        if lower_texts:
            self.lower_texts(text_col)
        if tokenize:
            self.tokenize_texts(text_col)
