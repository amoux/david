"""
david.text.tokenizers
~~~~~~~~~~~~~~~~~~~~~
This is where I will start adding various tokenizers and also
other classes that do not belog here at the momemnt but do
depend on each other. I Need to refactor a lot of files before
adding more files.
"""
from __future__ import print_function, unicode_literals

import os
import re
import string
import unicodedata
from typing import Dict, Iterator, List, NewType, Pattern, Tuple, Union

import spacy
import tensorflow as tf
import torch
from nltk.tokenize.casual import (EMOTICON_RE, HANG_RE, WORD_RE,
                                  _replace_html_entities, reduce_lengthening,
                                  remove_handles)

from ..lang import SPACY_STOP_WORDS, replace_contractions
from .prep import unicode_to_ascii

SpacyNlp = NewType("SpacyNlp", spacy.lang)


class SocialMediaTokenizer:
    """Social media text tokenizer.
    Adapted from NLTK's TweetTokenizer class.
    """

    def __init__(
        self,
        preserve_case: bool = True,
        reduce_len: bool = False,
        strip_handles: bool = False,
    ):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, sequence: str) -> List[str]:
        sequence: str = _replace_html_entities(sequence)
        if self.strip_handles:
            sequence = remove_handles(sequence)
        if self.reduce_len:
            sequence = reduce_lengthening(sequence)
        safe_seq = HANG_RE.sub(r"\1\1\1", sequence)
        words = WORD_RE.findall(safe_seq)
        if not self.preserve_case:
            words = list(
                map((lambda x: x if EMOTICON_RE.search(x)
                    else x.lower()), words)
            )
        return words


class VocabularyBase(object):
    sos_special_token: int = 0
    eos_special_token: int = 1

    def __init__(self, name: str = None):
        self.name = name
        self.word2index: Dict[int] = {}
        self.word2count: Dict[str, int] = {}
        self.index2word: {
            self.sos_special_token: "SOS",
            self.eos_special_token: "EOS",
        }
        self.num_words: int = 2  # count for both SOS and EOS tokens.

    def add_words_from_split(self, sentence: str) -> None:
        for word in sentence.split():
            self.add_word(word)

    def iter_words_from_tokenizer(self, word: str) -> None:
        self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
        else:
            self.word2count[word] += 1


class CharacterTokenenizer(VocabularyBase):
    STRING_CHARACTERS: str = string.ascii_letters + " .,;'"

    def __init__(self):
        super().__init__("CharTokenizerVocab")

    def get_character_id(self, character: str) -> Pattern:
        """Finds character index from STRING_LETTERS, e.g. "a" = 0"""
        return self.STRING_CHARACTERS.find(character)

    def character_to_tensor(self, character: str) -> torch.TensorType:
        """Turn a single character into a <1 x n_characters> Tensor"""
        char_size = 1
        if len(character) != char_size:
            raise ValueError(f"Letter size must be = 1, not: {len(character)}")

        num_characters = len(self.STRING_CHARACTERS)
        tensor = torch.zeros(char_size, num_characters)
        tensor[0][self.get_character_id(character)] = char_size
        return tensor

    def sequence_to_tensor(self, sequence: str) -> torch.TensorType:
        """Turn a string sequence into an array of one-hot char vectors."""
        char_size = 1
        sequence_size = len(sequence)
        num_letters = len(self.STRING_CHARACTERS)
        tensor = torch.zeros(sequence_size, char_size, num_letters)
        for i, char in enumerate(sequence):
            tensor[i][0][self.get_character_id(char)] = char_size
        return tensor


class SentenceTokenizer(VocabularyBase):
    """Sentence tokenizer built with spacy.

    Usage:
        >>> tokenizer = SentenceTokenizer()
        >>> text = ("Hello world's this is one sentence!. "
                    "Yes? and another sentence.")
        >>> sent_tokens = tokenizer.tokenize(text)
            ...
            ['<start> hello world s this is one sentence ! . <end>',
             '<start> yes ? <end>',
             '<start> and another sentence . <end>']
    """

    def __init__(self):
        super().__init__("SentTokenizerVocab")

    def pad_punctuation(
            self, sequence: str, special_tokens: bool = False) -> Pattern:
        """Padding punctuation with white spaces keeping the punctuation."""
        s = unicode_to_ascii(sequence.lower().strip())
        s = re.sub(r"([?.!,¿])", r" \1 ", s)
        s = re.sub(r'[" "]+', " ", s)
        s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
        s = s.rstrip().strip()
        if special_tokens:
            s = "<start> " + s + " <end>"
        return s

    def tokenize(
        self,
        sequence: Union[List[str], str],
        special_tokens: bool = True,
        lang: str = "en_core_web_sm",
    ) -> Iterator[List[str]]:
        """Basic sentence tokenizer with the option to add a <start> and an
        <end> special token to the sentence so that the model know when to
        start and stop predicting."""
        if isinstance(sequence, str):
            sequence = [sequence]
        nlp: SpacyNlp = spacy.load(lang)
        for doc in nlp.pipe(sequence):
            for sent in doc.sents:
                sent = sent.text.strip()
                if special_tokens:
                    yield self.pad_punctuation(sent, special_tokens=True)
                else:
                    yield sent
