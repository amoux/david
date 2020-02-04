"""
david.text._proto_tokenizers
----------------------------
This module is a temporary hidden module to all the tokenizers
until all are optimized properly (And simplified!). In the meantime
the true tokenizers module will hold working Tokenizers. One by one.
"""
from __future__ import print_function, unicode_literals

import os
import re
from collections import Counter
from string import ascii_letters
from typing import IO, Dict, Iterator, List, Optional, Union

import spacy
import torch
from nltk.tokenize.casual import (
    EMOTICON_RE,
    HANG_RE,
    WORD_RE,
    _replace_html_entities,
    reduce_lengthening,
    remove_handles,
)

from .prep import normalize_whitespace, unicode_to_ascii


class VocabularyBase(object):
    sos_special_token: int = 0
    eos_special_token: int = 1

    def __init__(self, name: str = None):
        self.name = name
        self.word2index: Dict[str, int] = {}
        self.word2count: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {
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


class CharacterTokenizer(VocabularyBase):
    STRING_CHARACTERS: str = ascii_letters + " .,;'"

    def __init__(self):
        super().__init__("CharTokenizerVocab")

    def get_character_id(self, character: str) -> int:
        """Finds character index from STRING_LETTERS, e.g. "a" = 0"""
        return self.STRING_CHARACTERS.find(character)

    def character_to_tensor(self, character: str):
        """Turn a single character into a <1 x n_characters> Tensor"""
        char_size = 1
        if len(character) != char_size:
            raise ValueError(f"Letter size must be = 1, not: {len(character)}")

        num_characters = len(self.STRING_CHARACTERS)
        tensor = torch.zeros(char_size, num_characters)
        tensor[0][self.get_character_id(character)] = char_size
        return tensor

    def word_to_tensor(self, sequence: str):
        """Turn a string sequence into an array of one-hot char vectors."""
        char_size = 1
        sequence_size = len(sequence)
        num_letters = len(self.STRING_CHARACTERS)
        tensor = torch.zeros(sequence_size, char_size, num_letters)
        for i, char in enumerate(sequence):
            tensor[i][0][self.get_character_id(char)] = char_size
        return tensor


class WordTokenizer(CharacterTokenizer):
    def __init__(
        self,
        preserve_case: bool = True,
        reduce_len: bool = False,
        strip_handles: bool = False,
    ):
        super()
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, sequence: str) -> List[str]:
        sequence = _replace_html_entities(sequence)
        if self.strip_handles:
            sequence = remove_handles(sequence)
        if self.reduce_len:
            sequence = reduce_lengthening(sequence)
        safe_seq = HANG_RE.sub(r"\1\1\1", sequence)
        words = WORD_RE.findall(safe_seq)
        if not self.preserve_case:
            emoji = EMOTICON_RE.search
            words = list(map((lambda w: w if emoji(w) else w.lower()), words))
        return words


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

    def pad_punctuation(self, sequence: str, special_tokens: bool = False):
        """Padding punctuation with white spaces keeping the punctuation."""
        string = unicode_to_ascii(sequence.lower().strip())
        string = re.sub(r"([?.!,¿])", r" \1 ", string)
        string = re.sub(r'[" "]+', " ", string)
        string = re.sub(r"[^a-zA-Z?.!,¿]+", " ", string)
        string = string.rstrip().strip()
        if special_tokens:
            string = "<start> " + string + " <end>"
        return string

    def tokenize(
        self,
        sequence: Union[List[str], str],
        special_tokens: bool = True,
        lang: str = "en_core_web_sm",
    ) -> Iterator[List[str]]:
        """Basic sentence tokenizer with the option to add a <start> and an
        <end> special token to the sentence so that the model know when to
        start and stop predicting.
        """
        if isinstance(sequence, str):
            sequence = [sequence]

        nlp = spacy.load(lang)
        for doc in nlp.pipe(sequence):
            for sent in doc.sents:
                sent = sent.text.strip()
                if special_tokens:
                    yield self.pad_punctuation(sent, special_tokens=True)
                else:
                    yield sent


class BaseTokenizer(object):
    """Base tokenizer class for all tokenizers."""

    def __init__(
        self,
        vocab_file: Optional["vocab.txt"] = None,
        document: Optional[List[str]] = None,
    ):
        """
        vocab_file: Either load an existing vocabulary of tokens.
        document: Or load from an iterable list of string sequences.
        preprocess: Normalize whitespace and enforce ASCII.
        tokenizer: Callable method. If None, WordTokenizer is used.
        """
        self.tokens_to_ids: Dict[str, int] = {}
        self.token_counter: Dict[str, int] = Counter()
        self._num_tokens: int = 0
        self._string_normalizer: object = None

        if vocab_file:
            self.from_file(vocab_file)
        elif document:
            self.from_doc(document)

    def add_token(self, token: Union[List[str], str]):
        """Add a single string token (word) to the vocabulary."""
        # A List[token] can be added only of theres only one string token.
        if isinstance(token, list) and len(token) == 1:
            token = str(token[0])

        if token not in self.tokens_to_ids:
            self.tokens_to_ids[token] = self._num_tokens
            self._num_tokens += 1
            self.token_counter[token] = 1
        else:
            self.token_counter[token] += 1

    def to_file(self, file_name="vocab.txt") -> IO:
        """Saves tokens to vocabulary text file."""
        with open(file_name, "w") as vocab_file:
            for token in self.tokens_to_ids.keys():
                vocab_file.write(f"{token}\n")

    def from_file(self, file_name="vocab.txt") -> IO:
        """Add tokens from a vocaulary text file."""
        with open(file_name, "r") as vocab_file:
            for token in vocab_file:
                self.add_token(token.replace("\n", ""))

    def from_doc(self, document: List[str]) -> None:
        """Add tokens from an iterable of string sequences."""
        for string in document:
            string = self._string_normalizer(string)
            tokens = self.tokenize(string)
            for token in tokens:
                self.add_token(token)

    def _encode(self, string: str) -> List[int]:
        """Converts a string in a sequence of integer ids using the tokenizer
        and vocabulary. NOTE: whitespace and ASCII normalization is applied.
        """
        tok2id = self.tokens_to_ids
        string = self._string_normalizer(string)
        tokens = self.tokenize(string)
        return [tok2id[token] for token in tokens if token in tok2id]

    def _decode(self, tokens: List[int]) -> List[str]:
        # Root vocab decoder, converts a sequence of toks[id] => toks[str]
        id2tok = {i: t for t, i in self.tokens_to_ids.items()}
        tokens = [id2tok[index] for index in tokens if index in id2tok]
        return tokens

    def convert_tokens_to_string(
        self, tokens: List[str], clean_tokenization=True
    ) -> str:
        """Convert a sequence of tokens (string) in a single string."""
        string = " ".join(tokens)
        if clean_tokenization:
            string = BaseTokenizer.clean_tokenization(string)
        return string

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts a single or a list of string tokens, in a single int:id."""
        tok2id = self.tokens_to_ids
        return [tok2id[token] for token in tokens if token in tok2id]

    def convert_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """Converts a single index or a sequence of indices (integers) in a token,
        using the vocabulary and tokens.
        """
        return self._decode(tokens)

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


class SocialMediaTokenizer(BaseTokenizer):
    r"""Social media aware tokenizer.

    Usage:
        >>> tokenizer = SocialMediaTokenizer('yt-comments-md')
        '< SocialMediaTokenizer(tokens=63844) >'
        >>> text = "Hello world! This is a social media tokenizer 🤗"
        >>> tokenized_text = tokenizer.tokenize(text)
        ['hello', 'world', '!', 'this', 'is', 'a', 'social', 'media',
            'tokenizer', '🤗']
        >>> indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        [477, 467, 164, 9, 49, 22, 3641, 4202, 1809]
        >>> tokenizer.convert_ids_to_tokens(indexed_tokens)
        ['hello', 'world', '!', 'this', 'is', 'a', 'social', 'media', '🤗']

        ## Adding a new token if its not in the vocabulary.
        >>> tokenizer.add_token("tokenizer") # or ["tokenizer"] of len=1
        >>> indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        [477, 467, 164, 9, 49, 22, 3641, 4202, 63844, 1809]
        ## The `token="tokenizer"` was added to `index=63844`.
        >>> tokenizer.convert_ids_to_tokens(indexed_tokens)
        ['hello', 'world', '!', 'this', 'is', 'a', 'social', 'media',
        'tokenizer', '🤗']

    """
    SM_TOKENIZER_MODELS = {"yt-comments-md": YTCommentsConfig.VOCAB_FILE}

    def __init__(
        self,
        vocab_file: Optional["vocab.txt"] = None,
        document: Optional[List[str]] = None,
        preserve_case=False,
        reduce_len=False,
        strip_handles=False,
    ):
        # This is not a real implementation!
        if vocab_file in self.SM_TOKENIZER_MODELS:
            vocab_file = self.SM_TOKENIZER_MODELS[vocab_file]
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles
        self._string_normalizer = self.normalize_string
        super().__init__(vocab_file, document)

    def normalize_string(self, string: str) -> str:
        if not self.preserve_case:
            string = string.lower()
        return normalize_whitespace(unicode_to_ascii(string))

    def tokenize(self, string: str) -> List[str]:
        string = _replace_html_entities(string)
        if self.strip_handles:
            string = remove_handles(string)
        if self.reduce_len:
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
        return f"< SocialMediaTokenizer(tokens={self._num_tokens}) >"
