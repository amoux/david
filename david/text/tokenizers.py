"""
David text tokenizer classes.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
from __future__ import print_function, unicode_literals

import copy
import json
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from string import ascii_letters
from typing import IO, Dict, List, Optional, Tuple, Union

import spacy
import torch
from nltk.tokenize.casual import (EMOTICON_RE, HANG_RE, WORD_RE,
                                  _replace_html_entities, reduce_lengthening,
                                  remove_handles)
from wasabi import msg

from .preprocessing import (clean_tokenization, normalize_whitespace,
                            remove_urls, string_printable, unicode_to_ascii)

WARN_INDEX_NOT_FREQ = (
    "\nWarning: Vocabulary's index has not been transformed to "
    "frequency.\nApplying the needed requirements for fitting the "
    "documents.\nCalling `self.vocab_index_to_frequency` for you...\n"
)

RECOMD_IO_LOADING = (
    "INFO: Using `self.save_vectors` and `self.load_vectors` is "
    "recommended over simply saving the vocabulary as it saves "
    "both states from the vocab_index and vocab_count dict objects "
    "Both which improve the tokenizer's features."
)


class TokenizerIO:
    """Vocab data object loader and writer for the tokenizer."""

    @staticmethod
    def save_obj(name: str, data_obj: object) -> IO:
        """Save the object as a pickle file."""
        with open(name, "wb") as file:
            pickle.dump(data_obj, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name: str) -> object:
        """Load the object from a pickle file."""
        with open(name, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def save_txt(name: str, dict_obj: Dict[str, int]) -> IO:
        """Save the indexed vocab to a vocab.txt file."""
        SEP = (",", ": ")
        with open(name, "w") as file:
            file.write(json.dumps(dict_obj, indent=4, separators=SEP))

    @staticmethod
    def load_txt(name: str) -> Dict[str, int]:
        """Load the indexed vocab from a vocab.txt file."""
        with open(name, "r") as file:
            return json.load(file)


class BaseTokenizer:
    """Base tokenization for all tokenizer classes."""

    def __init__(self):
        r"""Initialize tokenizer vocabulary.

        ## Subclassing configuration:

        - Construct a tokenizer callable method as `tokenize`:
            - `self.tokenize(sequence:str) -> str:`

        - Multiple ways for loading the vocabulary:
            - `BaseTokenizer.load_vocabulary(vocab_file: str = 'vocab.txt')`
            - `BaseTokenizer.load_vectors(vectors_file: str = 'vectors.pkl')`
            - `BaseTokenizer.fit_on_document(document: List[str])`

        """
        self.vocab_index: Dict[str, int] = {}
        self.vocab_count: Counter[Dict[str, int]] = Counter()
        self._token_count: int = 1
        self._index_vocab_is_frequency = False

    def add_token(self, token: Union[List[str], str]):
        """Add a single or more string sequences to the vocabulary."""
        if isinstance(token, list) and len(token) == 1:
            token = token[0]

        if token not in self.vocab_index:
            self.vocab_index[token] = self._token_count
            self._token_count += 1
            self.vocab_count[token] = 1
        else:
            self.vocab_count[token] += 1

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab_index)

    def bag_of_tokens(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return `n` [(token, token_id)] from the vocab index dict."""
        return list(self.vocab_index.items())[:n]

    def most_common(self, n: int = 5) -> List[Tuple[str, int]]:
        """Return `n` most common tokens in the vocabulary."""
        return self.vocab_count.most_common(n)

    def save_vocabulary(self, vocab_file="vocab.txt") -> IO:
        """Save the current vocabulary to a vocab.txt file."""
        msg.info(RECOMD_IO_LOADING)
        TokenizerIO.save_txt(vocab_file, self.vocab_index)

    def load_vocabulary(self, vocab_file="vocab.txt") -> IO:
        """Load the vocabulary from a vocab.txt file."""
        self.vocab_index = TokenizerIO.load_txt(vocab_file)

    def save_vectors(
        self, vectors_file="vectors.pkl", vectors: List[Tuple[str, int, int]] = None
    ) -> IO:
        """Save the vectors to a pickle file.
        
        vectors: (Optional) Pass a vectors object from `self.vocab_to_vectors()`
        or None, and the method is called automatically.
        """
        if vectors is None:
            vectors = self.vocab_to_vectors()
        TokenizerIO.save_obj(vectors_file, vectors)

    def load_vectors(self, vectors_file="vectors.pkl") -> IO:
        """Load both (index and count) vocab dicts from a vectors file.
        
        This method can be used as a way to restore both vocab and counter states.
        While `save_vocabulary` and `load_vocabulary` only save/load the
        self.vocab_index dictionary objects.
        """
        (vocab_index, vocab_count) = dict(), Counter()
        for token, count, token_id in TokenizerIO.load_obj(vectors_file):
            vocab_index[token] = token_id
            vocab_count[token] = count
        self.vocab_index, self.vocab_count = (vocab_index, vocab_count)

    def fit_on_document(self, document: List[str]):
        """Fit the vocabulary from an iterable document of string sequences."""
        for string in document:
            tokens = self.tokenize(string)
            for token in tokens:
                self.add_token(token)

    def vocab_to_vectors(self) -> List[Tuple[str, int, int]]:
        """Return the index-vocab as vector representation: `[(token, count, id)]`."""
        vectors = []
        vocfreq = self.vocab_count.most_common()
        for token_id, (token, count) in enumerate(vocfreq, start=1):
            vectors.append((token, count, token_id))
        return vectors

    def document_to_sequences(self, document: List[str]) -> List[int]:
        """Transform an iterable of string sequences to an iterable of intergers."""
        return list(self.document_to_sequences_generator(document))

    def document_to_sequences_generator(self, document: List[str]) -> List[int]:
        """Transform an iterable of string sequences to an iterable of intergers."""
        if not self._index_vocab_is_frequency:
            msg.warn(WARN_INDEX_NOT_FREQ)
            self.index_vocab_to_frequency()
            self._index_vocab_is_frequency = True

        for string in document:
            tokens = self.tokenize(string)
            if tokens is not None:
                yield self._encode(tokens)

    def index_vocab_to_frequency(self, inplace=True):
        """Align (Sort) the indexed vocabulary relative to the item(s) frequency.

        `inplace`: Weather to replace the existing `vocab_index` inplace if true.
            Otherwise, the dictionary `Dict[str, int]` is returned.
        """
        index_frequency: Dict[str, int] = {}
        vocab_index, _ = zip(*self.vocab_count.most_common())
        for index, token in enumerate(vocab_index, start=1):
            index_frequency[token] = index
        if inplace:
            self.vocab_index = index_frequency
            self._index_vocab_is_frequency = True
        else:
            return index_frequency

    def vectors_to_frequency(self, mincount=1):
        """Index vocabulary based on term frequency.

        mincount: Remove tokens with a count frequency of 1 or more.
            Default of 1 removes all uncommon tokens.
        """
        countmin = 0
        voc_size = len(self.vocab_index)
        freq_vocab_index = dict()
        freq_vocab_count = dict()
        vocab_count = copy.copy(self.vocab_count)
        vocab_count = sorted(vocab_count.items(), key=lambda x: x[1])
        for index, (token, count) in enumerate(vocab_count, start=1):
            if count > mincount:
                freq_vocab_index[token] = index
                freq_vocab_count[token] = count
            else:
                countmin += 1

        self.vocab_index = freq_vocab_index
        self.vocab_count = freq_vocab_count
        self._index_vocab_is_frequency = True
        del vocab_count
        del freq_vocab_index
        del freq_vocab_count
        msg.info(f"* Removed {voc_size - countmin} from original size {voc_size}")

    def _encode(self, tokens: List[str]) -> List[int]:
        tok2id = self.vocab_index
        token_ids = [tok2id[token] for token in tokens if token in tok2id]
        return token_ids

    def _decode(self, token_ids: List[int]) -> List[str]:
        id2tok = {idx: tok for tok, idx in self.vocab_index.items()}
        tokens = [id2tok[index] for index in token_ids if index in id2tok]
        return tokens

    def convert_string_to_tokens(self, string: str) -> List[str]:
        """Covert string to a sequence of string tokens.

        This is the same as using `self.tokenize(string)`.
        """
        tokens = self.tokenize(string)
        return tokens

    def convert_string_to_ids(self, string: str) -> List[int]:
        """Convert a string to a sequence of integer token ids."""
        tokens = self.tokenize(string)
        token_ids = self._encode(tokens)
        return token_ids

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of string tokens to a single string."""
        string = clean_tokenization(" ".join(tokens))
        return string

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert a sequence of string tokens to tokens of ids."""
        token_ids = self._encode(tokens)
        return token_ids

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert a sequence of integer ids to tokens of strings."""
        tokens = self._decode(token_ids)
        return tokens

    def convert_ids_to_string(self, token_ids: List[int]) -> str:
        """Convert a sequence of integer ids to a single string."""
        tokens = self._decode(token_ids)
        string = clean_tokenization(" ".join(tokens))
        return string


class Tokenizer(BaseTokenizer):
    """Core tokenizer class for processing social media texts."""

    def __init__(
        self,
        vectors_file: str = None,
        vocab_file: str = None,
        document: List[str] = None,
        remove_urls: bool = True,
        enforce_ascii: bool = True,
        preserve_case: bool = False,
        reduce_length: bool = False,
        strip_handles: bool = False,
    ):
        """Tokenizer for social media text."""
        super().__init__()
        self.remove_urls = remove_urls
        self.enforce_ascii = enforce_ascii
        self.preserve_case = preserve_case
        self.reduce_length = reduce_length
        self.strip_handles = strip_handles

        if vectors_file is not None:
            self.load_vectors(vectors_file)
        if vocab_file is not None:
            self.load_vocabulary(vocab_file)
        if document is not None:
            self.fit_on_document(document)

    def preprocess_string(self, string: str) -> str:
        """Normalize strings sequences to valid ASCII and printable format."""
        if not self.preserve_case:
            string = string.lower()
        if self.remove_urls:
            string = remove_urls(string)
        if self.enforce_ascii:
            string = string_printable(string)
        if self.strip_handles:
            string = remove_handles(string)
        if self.reduce_length:
            string = reduce_lengthening(string)
        string = _replace_html_entities(string)
        return normalize_whitespace(unicode_to_ascii(string))

    def tokenize(self, string: str) -> List[str]:
        """Tokenize a sequence of string characters."""
        string = self.preprocess_string(string)
        safe_string = HANG_RE.sub(r"\1\1\1", string)
        tokens = WORD_RE.findall(safe_string)
        if not self.preserve_case:
            emoji = EMOTICON_RE.search
            tokens = list(
                map((lambda token: token if emoji(token) else token.lower()), tokens)
            )
        return tokens

    def __repr__(self):
        """Return the size of the vocabulary in string format."""
        return f"<Tokenizer(vocab_size={self.vocab_size})>"
