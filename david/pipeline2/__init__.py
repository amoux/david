import re
from typing import Generator, List, Set, Union

import numpy as np
import spacy
from spacy.tokens import Doc


def avg_word_len(doc: object) -> float:
    lengths: List[int] = list()
    for token in doc:
        if token.is_alpha and not token.is_stop:
            lengths.append(len(token))
    if len(lengths) == 0:
        return 0.0
    return np.mean(lengths)


def count_stop_tokens(doc: object) -> int:
    count = 0
    for token in doc:
        if token.is_stop:
            count += 1
    return count


def count_digit_tokens(doc: object) -> int:
    count = 0
    for token in doc:
        if token.is_digit:
            count += 1
    return count


def count_word_tokens(doc: object) -> int:
    count = 0
    for token in doc:
        if not token.is_stop and not token.is_digit and not token.is_punct:
            count += 1
    return count


def count_upper_chars(doc: object) -> int:
    count = 0
    for token in doc:
        upper_chars = re.findall(r"[A-Z]", token.text)
        if upper_chars:
            count += len(upper_chars)
    return count


def count_lower_chars(doc: object) -> int:
    count = 0
    for token in doc:
        lower_chars = re.findall(r"[a-z]", token.text)
        if lower_chars:
            count += len(lower_chars)
    return count


class SpacyDocMetric(object):
    name = "doc_metric"

    def __init__(self, nlp, force=True):
        self.GETTERS = {
            "avg_word_len": avg_word_len,
            "count_word_tokens": count_word_tokens,
            "count_stop_tokens": count_stop_tokens,
            "count_digit_tokens": count_digit_tokens,
            "count_upper_chars": count_upper_chars,
            "count_lower_chars": count_lower_chars,
        }

        for name, getter in self.GETTERS.items():
            Doc.set_extension(name, getter=getter, force=force)


class DavidPipeline(SpacyDocMetric):
    def __init__(self, nlp, force=True):
        super().__init__(nlp=nlp, force=force)

    def __call__(self, batch: List[str], batch_size: int = 50) -> Generator:
        for doc in nlp.pipe(batch, batch_size=batch_size):
            yield doc

    def update_stop_words(
            self, stop_words: Union[List[str], Set[str]]) -> None:
        nlp.Defaults.stop_words.update(stop_words)
        for word in stop_words:
            lexemme = nlp.vocab[word]
            lexemme.is_stop = True
