"""
David text tokenizer classes.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is where I will start adding various tokenizers and also
other classes that do not belog here at the momemnt but do
depend on each other. I Need to refactor a lot of files before
adding more files.
"""
from __future__ import print_function, unicode_literals

import os
import re
from collections import Counter
from pathlib import Path
from string import ascii_letters
from typing import IO, Dict, Iterator, List, Optional, Union

import pandas as pd
import spacy
import torch
from nltk.tokenize.casual import (EMOTICON_RE, HANG_RE, WORD_RE,
                                  _replace_html_entities, reduce_lengthening,
                                  remove_handles)

from .prep import normalize_whitespace, unicode_to_ascii


class YoutubeWebMD:
    """Temporary `configuration` while prototyping."""

    VOCAB_PATH = "/home/ego/david_data/vocab/yt_web_md/"
    VOCAB_FILE = os.path.join(VOCAB_PATH, "vocab.txt")
    CSV_CORPUS = os.path.join(VOCAB_PATH, "corpus.csv")
    model_name = "yt-web-md"

    @staticmethod
    def load_corpus_from_df() -> Union[pd.DataFrame, None]:
        """Load the corpus and returns a `pandas.Dataframe` instance."""
        return pd.read_csv(YoutubeWebMD.CSV_CORPUS)

    @staticmethod
    def load_corpus_from_as_doc() -> List[str]:
        """Return an generator of iterable string sequences."""
        df = YoutubeWebMD.load_corpus_from_df()
        for sequence in df["text"].tolist():
            yield sequence


class BaseTokenizer(object):
    """Base tokenization for all tokenizer classes."""

    def __init__(self):
        r"""When subclassing the base tokenizer.

        * The following methods are expected (Example):
        ```python
        class ChildTokenizer(BaseTokenizer):
            def __init__(self, *args, **kwargs):
                super().__init__()
            self.__string_normalizer = self.my_normalizer_method
        ...
            def my_normalizer_method(self, text:str): -> str:
        ```
        - Configure how the vocab will be loaded; use the following methods:
            - `BaseTokenizer.vocab_from_file(vocab_file='<path/to/vocab.txt>')`
            - `BaseTokenizer.vocab_from_doc(List[str])`

        """
        self.tokens_to_ids: Dict[str, int] = {}
        self.token_counts: Dict[str, int] = {}
        self.num_tokens = 0
        self._string_normalizer: object = None

    def add_token(self, token: Union[List[str], str]):
        """Add a single or more string sequences to the vocabulary."""
        if isinstance(token, list) and len(token) == 1:
            token = str(token[0])
        if token not in self.tokens_to_ids:
            self.tokens_to_ids[token] = self.num_tokens
            self.num_tokens += 1
            self.token_counts[token] = 1
        else:
            self.token_counts[token] += 1

    def save_vocabulary(self, file_name="vocab.txt") -> IO:
        """Save the current vocabulary to a vocab.txt file."""
        with open(file_name, "w") as vocab_file:
            for token in self.tokens_to_ids.keys():
                vocab_file.write(f"{token}\n")

    def vocab_from_file(self, file_name="vocab.txt") -> IO:
        """Load the vocabulary from a vocab.txt file."""
        with open(file_name, "r") as vocab_file:
            for token in vocab_file:
                self.add_token(token.replace("\n", ""))

    def vocab_from_doc(self, document: List[str]):
        """Load the vocabulary from a document of strings."""
        for string in document:
            string = self._string_normalizer(string)
            tokens = self.tokenize(string)
            for token in tokens:
                self.add_token(token)

    def __vocabulary_encoder(self, tokens: List[str]) -> List[int]:
        tok2id = self.tokens_to_ids
        tokens = [tok2id[token] for token in tokens if token in tok2id]
        return tokens

    def __vocabulary_decoder(self, tokens: List[int]) -> List[str]:
        id2tok = {i: t for t, i in self.tokens_to_ids.items()}
        tokens = [id2tok[index] for index in tokens if index in id2tok]
        return tokens

    def convert_string_to_tokens(self, string: str) -> List[str]:
        """Covert string to a sequence of string tokens.

        This method is the same as calling `self.tokenize('string')`.
        """
        tokens = self.tokenize(self._string_normalizer(string))
        return tokens

    def convert_string_to_ids(self, string: str) -> List[int]:
        """Convert a string to a sequence of integer token ids."""
        tokens = self.convert_string_to_tokens(string)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of string tokens to a single string."""
        string = BaseTokenizer.clean_tokenization(" ".join(tokens))
        return string

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert a sequence of string tokens to tokens of ids."""
        token_ids = self.__vocabulary_encoder(tokens)
        return token_ids

    def convert_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """Convert a sequence of integer ids to tokens of strings."""
        tokens = self.__vocabulary_decoder(tokens)
        return tokens

    def convert_ids_to_string(self, tokens: List[int]) -> str:
        """Convert a sequence of integer ids to a single string."""
        tokens = self.convert_ids_to_tokens(tokens)
        string = self.convert_tokens_to_string(tokens)
        return string

    @staticmethod
    def clean_tokenization(string: str) -> str:
        """Clean up spaces before punctuations and abreviated forms."""
        string = (
            string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
            .replace(" / ", "/")
        )
        return string


class WordTokenizer(BaseTokenizer):
    """Word tokenizer class with social media aware context."""

    # Toy "model" for prototyping with real data :)
    MODELS = {"yt-web-md": YoutubeWebMD.VOCAB_FILE}
    # Recommended tokenizer defaults.
    preserve_case: bool = False
    reduce_length: bool = False
    strip_handles: bool = False

    def __init__(
        self,
        vocab_file: Optional["vocab.txt"] = None,
        document: Optional[List[str]] = None,
    ):
        """Word tokenizer with social media aware contenxt."""
        super().__init__()
        self._string_normalizer: object = self.normalize_string

        if vocab_file and document is None:
            if vocab_file.startswith("yt") or vocab_file in self.MODELS.keys():
                vocab_file_from_pretrained = self.MODELS[vocab_file]
                self.vocab_from_file(vocab_file_from_pretrained)
            elif os.path.isfile(vocab_file):
                self.vocab_from_file(vocab_file)
        elif document and vocab_file is None:
            self.vocab_from_doc(document)
        else:
            raise Exception(
                "Vocabulary could not be loaded from {}".format(vocab_file or document)
            )

    def normalize_string(self, string: str) -> str:
        """Normalize strings by encoding ASCII and excessive whitespaces."""
        if not self.preserve_case:
            string = string.lower()
        return normalize_whitespace(unicode_to_ascii(string))

    def tokenize(self, string: str) -> List[str]:
        """Tokenize a sequence of string characters."""
        string = _replace_html_entities(string)
        if self.strip_handles:
            string = remove_handles(string)
        if self.reduce_length:
            string = reduce_lengthening(string)

        safe_string = HANG_RE.sub(r"\1\1\1", string)
        tokens = WORD_RE.findall(safe_string)
        if not self.preserve_case:
            emoji = EMOTICON_RE.search
            tokens = list(
                map((lambda token: token if emoji(token) else token.lower()), tokens)
            )
        return tokens

    def __repr__(self):
        """Return the size of the vocabulary in string format."""  # why do i need to add docs PEP8?
        return f"< WordTokenizer(vocab_size={self.num_tokens}) >"
