import copy
import re
from collections import MutableSequence
from typing import Any, Dict, Iterable, List, NoReturn, Optional, Set, Tuple, Union

import numpy as np
from pandas.api.types import CategoricalDtype

from ..text.preprocessing import (
    extract_emojis,
    get_sentiment_polarity,
    get_sentiment_subjectivity,
    normalize_whitespace,
)
from ..text.utils import change_case

# Regex patterns used. (replacing these soon)
TIME_RE = r"(\d{1,2}\:\d{1,2})"
URL_RE = r"(http\S+)"
TAG_RE = r"(\#\w+)"


DEFAULT_COLUMNS = {
    "string": ["stringLength"],
    "word": ["avgWordLength", "stopWordCount", "wordCount"],
    "char": ["charDigitCount", "charUpperCount", "charLowerCount"],
    "senti": ["polarityScore", "subjectivityScore", "sentiLabel"],
    "author": ["authorTimeTag", "authorUrl", "authorHashTag", "authorEmoji"],
}


def avg_word_length(sequence: str, stop_words: Union[List[str], Set[str]]) -> float:
    word_lengths = [len(w) for w in sequence.split() if w not in stop_words]
    if len(word_lengths) == 0:
        return 0.0
    return np.mean(word_lengths)


def count_stop_words(sequence: str, stop_words: Union[List[str], Set[str]]) -> int:
    return len([w for w in sequence.split() if w in stop_words])


def count_true_words(sequence: str, stop_words: Union[List[str], Set[str]]) -> int:
    return len([w for w in sequence.split() if w not in stop_words])


def count_upper_chars(sequence: str) -> int:
    return len(re.findall(r"[A-Z]", sequence))


def count_lower_chars(sequence: str) -> int:
    return len(re.findall(r"[a-z]", sequence))


def count_digit_chars(sequence: str) -> int:
    return len(re.findall(r"[0-9]", sequence))


def count_words(sequence: str) -> int:
    return len(sequence)


def columns_to_snakecase(cols: Dict[str, List[str]]) -> Dict[str, List[str]]:
    cols = copy.copy(cols)
    for key, col in cols.items():
        snakecased = [change_case(name) for name in col]
        cols[key].clear()
        cols[key].extend(snakecased)
    return cols


class TextMetrics(MutableSequence, object):

    _DEFAULT_COLS: Dict[str, List[str]] = DEFAULT_COLUMNS
    COLUMNS = {"string": [], "word": [], "char": [], "senti": [], "author": []}
    SENTI_LABELS = ["positive", "negative", "neutral"]

    def _load_metric_columns(self, to_snake: bool = False) -> None:
        if to_snake:
            self._DEFAULT_COLS = columns_to_snakecase(DEFAULT_COLUMNS)
        else:
            self._DEFAULT_COLS = DEFAULT_COLUMNS

    def _update_columns(self, key: str) -> None:
        self.COLUMNS[key].clear()
        self.COLUMNS[key].extend(self._DEFAULT_COLS[key])

    def sentiment_label(self, score: float) -> Any:
        if score > 0:
            return self.SENTI_LABELS[0]
        if score < 0:
            return self.SENTI_LABELS[1]
        else:
            return self.SENTI_LABELS[2]

    def string_metric(self, text_col: str = "text") -> None:
        strcol = self._DEFAULT_COLS["string"]
        self[text_col] = self[text_col].apply(lambda w: normalize_whitespace(w))
        self[strcol[0]] = self[text_col].apply(lambda w: count_words(w))
        self._update_columns("string")

    def word_metric(self, text_col: str = "text") -> None:
        wordcol = self._DEFAULT_COLS["word"]
        self[wordcol[0]] = self[text_col].apply(
            lambda w: avg_word_length(w, self.STOP_WORDS)
        )
        self[wordcol[1]] = self[text_col].apply(
            lambda w: count_stop_words(w, self.STOP_WORDS)
        )
        self[wordcol[2]] = self[text_col].apply(
            lambda w: count_true_words(w, self.STOP_WORDS)
        )
        self._update_columns("word")

    def char_metric(self, text_col: str = "text") -> None:
        chars = self._DEFAULT_COLS["char"]
        self[chars[0]] = self[text_col].apply(lambda w: count_digit_chars(w))
        self[chars[1]] = self[text_col].apply(lambda w: count_upper_chars(w))
        self[chars[2]] = self[text_col].apply(lambda w: count_lower_chars(w))
        self._update_columns("char")

    def senti_metric(self, text_col: str = "text") -> None:
        senti_cols = self._DEFAULT_COLS["senti"]
        self[senti_cols[0]] = self[text_col].apply(lambda w: get_sentiment_polarity(w))
        self[senti_cols[1]] = self[text_col].apply(
            lambda w: get_sentiment_subjectivity(w)
        )
        # Assing sentiment labels as a categorical data type.
        cat_dtype = CategoricalDtype(self.SENTI_LABELS, ordered=False)
        self[senti_cols[2]] = (
            self[senti_cols[0]]
            .apply(lambda w: self.sentiment_label(w))
            .astype(cat_dtype)
        )
        self._update_columns("senti")

    def author_metric(self, text_col: str = "text") -> None:
        """Extract author tags.

        - time-tag  : extracts video time tags, e.g. 1:20.
        - url-link  : extracts urls links if found.
        - hash-tag  : extracts hash tags, e.g. #numberOne.
        - emojis    : extracts emojis  👾.
        """
        AUTHOR_COLS = self._DEFAULT_COLS["author"]
        self[AUTHOR_COLS[0]] = self[text_col].str.extract(TIME_RE)
        self[AUTHOR_COLS[1]] = self[text_col].str.extract(URL_RE)
        self[AUTHOR_COLS[2]] = self[text_col].str.extract(TAG_RE)
        self[AUTHOR_COLS[3]] = self[text_col].apply(lambda x: extract_emojis(x))
        self._update_columns("author")

    def load_text_metrics(
        self,
        text_col: str = "text",
        string: bool = True,
        words: bool = True,
        chars: bool = True,
        senti: bool = False,
        author: bool = False,
        snakecased: bool = False,
        stop_words: Optional[Union[List[str], Set[str]]] = None,
        sentilabels: Optional[Union[List[str], Tuple[str]]] = None,
    ) -> NoReturn:
        """Single function call to extract standard information from sequences.

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
        self._load_metric_columns(snakecased)
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
        return f"{self.info()}"
