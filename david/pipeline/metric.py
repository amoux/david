
import copy
from collections import MutableSequence
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from pandas.api.types import CategoricalDtype

from ..text.prep import (extract_emojis, get_sentiment_polarity,
                         get_sentiment_subjectivity, normalize_whitespace)
from ..text.utils import change_case

# Regex patterns used. (replacing these soon)
TIME_RE = r"(\d{1,2}\:\d{1,2})"
URL_RE = r"(http\S+)"
TAG_RE = r"(\#\w+)"


DEFAULT_COLUMNS = {
    'string': ['stringLength'],
    'word': ['avgWordLength', 'stopWordCount', 'wordCount'],
    'char': ['charDigitCount', 'charUpperCount', 'charLowerCount'],
    'senti': ['polarityScore', 'subjectivityScore', 'sentiLabel'],
    'author': ['authorTimeTag', 'authorUrl', 'authorHashTag', 'authorEmoji']
}


def columns_to_snakecase(cols: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cols = copy.copy(cols)
    for key, col in cols.items():
        snakecased = [change_case(name) for name in col]
        cols[key].clear()
        cols[key].extend(snakecased)
    return cols


class TextMetrics(MutableSequence, object):
    _DEFAULT_COLS: Dict[str, List[str]] = None
    COLUMNS = {'string': [], 'word': [], 'char': [], 'senti': [], 'author': []}
    SENTI_LABELS = ['positive', 'negative', 'neutral']

    def _columns_case_type(self, to_snake: bool = False):
        if to_snake:
            self._DEFAULT_COLS = columns_to_snakecase(DEFAULT_COLUMNS)
        else:
            self._DEFAULT_COLS = DEFAULT_COLUMNS

    def _update_columns(self, key: str):
        self.COLUMNS[key].clear()
        self.COLUMNS[key].extend(self._DEFAULT_COLS[key])

    def sentiment_label(self, score: float) -> Any:
        if score > 0:
            return self.SENTI_LABELS[0]
        if score < 0:
            return self.SENTI_LABELS[1]
        else:
            return self.SENTI_LABELS[2]

    def avg_words(self, words: list) -> List[float]:
        return [len(w) for w in words.split(' ') if w not in self.STOP_WORDS]

    def string_metric(self, text_col='text') -> None:
        STRING_COLS = self._DEFAULT_COLS['string']
        self[text_col] = self[text_col].apply(
            lambda x: normalize_whitespace(x))
        self[STRING_COLS[0]] = self[text_col].str.len()
        self._update_columns('string')

    def word_metric(self, text_col='text') -> None:
        WORD_COLS = self._DEFAULT_COLS['word']
        self[WORD_COLS[0]] = self[text_col].apply(
            lambda x: np.mean(
                self.avg_words(x)) if len(self.avg_words(x)) > 0 else 0)
        self[WORD_COLS[1]] = self[text_col].apply(
            lambda x: len(
                [w for w in x.split(" ") if w in self.STOP_WORDS]))
        self[WORD_COLS[2]] = self[text_col].apply(
            lambda x: len(
                [w for w in x.split(" ") if w not in self.STOP_WORDS]))
        self._update_columns('word')

    def char_metric(self, text_col='text') -> None:
        CHAR_COLS = self._DEFAULT_COLS['char']
        self[CHAR_COLS[0]] = self[text_col].str.findall(r'[0-9]').str.len()
        self[CHAR_COLS[1]] = self[text_col].str.findall(r'[A-Z]').str.len()
        self[CHAR_COLS[2]] = self[text_col].str.findall(r'[a-z]').str.len()
        self._update_columns('char')

    def senti_metric(self, text_col='text') -> None:
        SENTI_COLS = self._DEFAULT_COLS['senti']
        self[SENTI_COLS[0]] = self[text_col].apply(
            lambda x: get_sentiment_polarity(x))
        self[SENTI_COLS[1]] = self[text_col].apply(
            lambda x: get_sentiment_subjectivity(x))
        # Assing sentiment labels as a categorical data type.
        cat_dtype = CategoricalDtype(self.SENTI_LABELS, ordered=False)
        self[SENTI_COLS[2]] = self[SENTI_COLS[0]].apply(
            lambda x: self.sentiment_label(x)).astype(cat_dtype)
        self._update_columns('senti')

    def author_metric(self, text_col='text') -> None:
        AUTHOR_COLS = self._DEFAULT_COLS['author']
        self[AUTHOR_COLS[0]] = self[text_col].str.extract(TIME_RE)
        self[AUTHOR_COLS[1]] = self[text_col].str.extract(URL_RE)
        self[AUTHOR_COLS[2]] = self[text_col].str.extract(TAG_RE)
        self[AUTHOR_COLS[3]] = self[text_col].apply(
            lambda x: extract_emojis(x))
        self._update_columns('author')

    def load_text_metrics(
            self,
            text_col: str = 'text',
            string: bool = True,
            words: bool = True,
            chars: bool = True,
            senti: bool = False,
            author: bool = False,
            snakecased: bool = False,
            stop_words: Optional[Union[List[str], Set[str]]] = None,
            sentilabels: Optional[Union[List[str], Tuple[str]]] = None,
    ) -> None:
        """Single function call to extract standard information from sequences.

        Parameters:
        ----------

        `stopwords` (Optional[Union[List[str], Set[str]]], default=None):
            default, STOP_WORDS=spacy.lang.en.STOP_WORDS
            Used to capture metrics of actual words and stopwords
            detected within a string sequence.

        `sentilabels` (Optional[Union[List[str], Tuple[str]]], default=None):
            default, SENTI_LABELS=('positive', 'negative', 'neutral')
            Used to label a text for every index in a column from a
            table containing sentiment values. Pass a tuple to change
            the default labels to use. e.g. (1, 0, 'N')

        `snakecased` (bool, default=False):
            If true, the names of any metric column will be in `snake_case`
            style. Otherwise, the default `camelCase` style will be used.
            Use the class attribute `self.COLUMNS` to quickly access the
            column names.

        """
        self._columns_case_type(snakecased)
        if stop_words is not None:
            self.STOP_WORDS = stop_words
        if sentilabels is not None:
            self.SENTI_LABELS = sentilabels
        if string:
            self.string_metric(text_col)
        if words:
            self.word_metric(text_col)
        if chars:
            self.char_metric(text_col)
        if senti:
            self.senti_metric(text_col)
        if author:
            self.author_metric(text_col)

    def __repr__(self):
        return f'{ type(self) }'
