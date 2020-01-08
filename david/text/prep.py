from __future__ import unicode_literals

import collections
import re
import string
import unicodedata
from typing import Generator, List

import emoji
import gensim
import nltk
import pattern
import spacy
import textblob
from nltk.tokenize.casual import (EMOTICON_RE, HANG_RE, WORD_RE,
                                  _replace_html_entities, reduce_lengthening,
                                  remove_handles)

from ..lang import SPACY_STOP_WORDS, replace_contractions


class YTCommentTokenizer:
    """Youtube comment tokenizer, adapted from NLTK's TweetTokenizer class."""

    def __init__(self, preserve_case=True, reduce_len=False,
                 strip_handles=False):
        self.preserve_case = preserve_case
        self.reduce_len = reduce_len
        self.strip_handles = strip_handles

    def tokenize(self, sequence):
        sequence = _replace_html_entities(sequence)
        if self.strip_handles:
            sequence = remove_handles(sequence)
        if self.reduce_len:
            sequence = reduce_lengthening(sequence)
        safe_seq = HANG_RE.sub(r"\1\1\1", sequence)
        words = WORD_RE.findall(safe_seq)
        if not self.preserve_case:
            words = list(map((
                lambda x: x if EMOTICON_RE.search(x) else x.lower()), words))
        return words


def unicode_to_ascii(sequence: str) -> str:
    """Convert the unicode to ASCII."""
    return "".join(char for char in unicodedata.normalize("NFD", sequence)
                   if unicodedata.category(char) != "Mn")


def extract_emojis(sequence: str):
    """Extracts all emoji characters found in a sequence."""
    return " ".join([char for char in sequence if char in emoji.UNICODE_EMOJI])


def get_sentiment_polarity(sequence: str):
    # NOTE: Replacing all Textblob methods with optimized sentiment models.
    return textblob.TextBlob(sequence).sentiment.polarity


def get_sentiment_subjectivity(sequence: str):
    # NOTE: Replacing all Textblob methods with optimized sentiment models.
    return textblob.TextBlob(sequence).sentiment.subjectivity


def lemmatizer(doc: list):
    Lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([Lemmatizer.lemmatize(sent) for sent in doc])


def spacy_token_lemmatizer(tokens, postags=None):
    """SpaCy's part-of-speech lemmatizer for tokens"""
    if not postags:
        postags = ["NOUN", "ADJ", "VERB", "ADV"]

    nlp = spacy.load("en_core_web_sm", disable=("parser", "ner"))
    lemmas = list()
    for sequence in tokens:
        sequence = nlp(" ".join(sequence))
        lemmas.append(
            [tok.lemma_ for tok in sequence if tok.pos_ in postags])
    return lemmas


def spacy_sentence_tokenizer(doc: list):
    """Spacy sentence tokenizer."""
    nlp = spacy.load("en_core_web_lg")
    sents = list()
    for line in doc:
        doc = nlp(line)
        for sent in doc.sents:
            sents.append(sent.text)
    return sents


def sentence_tokenizer(doc: List[str]) -> Generator:
    """Replacing `spacy_sentence_tokenizer` with this generator."""
    nlp = spacy.load("en_core_web_sm")
    for line in doc:
        line = nlp(normalize_whitespace(line))
        for sent in line.sents:
            yield sent.text


def treebank_to_wordnet_pos(pos_tag: str):
    """NLTK POS-tagger, converts penn treebank tags wordnet tags."""
    if pos_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif pos_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif pos_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif pos_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return None


def part_of_speech_annotator(sequence: str):
    """Annotates text tokens with pos tags, uses NLTK's Wordnet POS."""
    tagged_seq = pattern.text.en.tag(sequence)
    return [(word.lower(), treebank_to_wordnet_pos(pos_tag))
            for word, pos_tag in tagged_seq]


def part_of_speech_lemmatizer(sequence: str):
    """Lemmatize sequences based on part-of-speech tags."""
    Lemmatizer = nltk.stem.WordNetLemmatizer()
    tagged_seq = part_of_speech_annotator(sequence)
    return " ".join([Lemmatizer.lemmatize(word, pos) if pos else word
                     for word, pos in tagged_seq])


def normalize_whitespace(sequence: str):
    LINEBREAK = re.compile(r"(\r\n|[\n\v])+")
    NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)
    return NONBREAKING_SPACE.sub(" ", LINEBREAK.sub(r"\n", sequence)).strip()


def normalize_wiggles(sequence, min_words=10):
    """Normalizes wiggles from sequences.

    What is a 'wiggle'?  A wiggle is one - two - tree words that are repeated
    over and over by social media users e.g, `hello hello hello hello X 1000.`
    This method properly normalizes sequences by using top most frequent word
    over the whole sequence. It uses the collections.Counter class for speed.

    Why the name? A wiggle is the word a youtube user used to post a comment
    repeating wiggle over 1000 times and its the reason why I came up for this
    method - because I wanted to remove the damn wiggle wiggle from my dataset!
    """
    if len(sequence.split()) < min_words:
        return sequence
    tokens, _ = zip(*collections.Counter(sequence.split()).most_common())
    if tokens[0] in sequence.split():
        return " ".join(tokens)
    return sequence


def remove_repeating_words(sequence: str):
    return " ".join(collections.Counter(
        [tok for tok in sequence.split()]).keys())


def remove_repeating_characters(sequence: str):
    """Removes repeating characters from one/multiple words in a sequence."""

    def normalize(sequence):
        if nltk.corpus.wordnet.synsets(sequence):
            return sequence
        repeat_pattern = re.compile(r"(\w*)(\w)\2(\w*)")
        sub_seq = repeat_pattern.sub(r"\1\2\3", sequence)
        return normalize(sub_seq) if sub_seq != sequence else sub_seq

    # if more than one word then normalize each word found in the sequence.
    if len(sequence.split()) > 1:
        return " ".join(normalize(word) for word in sequence.split())
    return normalize(sequence)


def remove_punctuation(sequence: str, contractions: bool = False) -> str:
    punctuation = re.escape(string.punctuation)
    punctuation = re.compile("[{}]".format(punctuation))
    # Removing contractions improves removing punctuation without
    # hurting the meaning of a word e.g, "y'all" => "you are all"
    # should be done before actually removing all punctuation.
    # NOTE that punctuations "-" and "_" are not removed.
    if contractions:
        sequence = replace_contractions(sequence)
    sequence = nltk_word_tokenizer(sequence)
    sequence = " ".join(filter(lambda tok: punctuation.sub("", tok), sequence))
    sequence = normalize_whitespace(sequence)
    return sequence


def remove_stopwords(sequence: str, stop_words=None):
    if not stop_words:
        stop_words = SPACY_STOP_WORDS
    tokens = nltk_word_tokenizer(sequence)
    sequence = " ".join([tok for tok in tokens if tok not in stop_words])
    return sequence.strip()


def nltk_word_tokenizer(sequence: str):
    """NLTK sequence and punctuation tokenizer.

    Usage:
        >>> nltk_word_tokenizer("Hello, how are you?")
        '['Hello', ',', 'how', 'are', 'you', '?']'
    """
    tokens = nltk.word_tokenize(sequence)
    return [tok.strip() for tok in tokens]


def preprocess_sequence(sequence: str,
                        contractions=True,
                        lemmatize=False,
                        punctuation=True,
                        norm_chars=True,
                        rm_stopwords=True,
                        stop_words=None,
                        tokenize=False):
    """Basic text preprocessing for a sequence."""
    sequence = normalize_whitespace(unicode_to_ascii(sequence))
    if contractions:
        sequence = replace_contractions(sequence)
    if lemmatize:
        sequence = part_of_speech_lemmatizer(sequence)
    if punctuation:
        sequence = remove_punctuation(sequence)
    if norm_chars:
        sequence = remove_repeating_characters(sequence)
    if rm_stopwords:
        sequence = remove_stopwords(sequence, stop_words=stop_words)
    if tokenize:
        sequence = nltk_word_tokenizer(sequence)
    return sequence


def preprocess_doc(doc: list,
                   contractions=True,
                   lemmatize=False,
                   punctuation=True,
                   norm_chars=True,
                   rm_stopwords=True,
                   stop_words=None,
                   tokenize=False):
    """Basic text preprocessing for a doc (iterable) of sequences."""
    for sequence in doc:
        yield preprocess_sequence(
            sequence=sequence,
            contractions=contractions,
            lemmatize=lemmatize,
            punctuation=punctuation,
            norm_chars=norm_chars,
            rm_stopwords=rm_stopwords,
            stop_words=stop_words,
            tokenize=tokenize,
        )


def gensim_preprocess(doc: list, stop_words=None, deacc=False):
    """Convert a document into a list of lowercase tokens.

    Ignores tokens that are too short or too long. and removes
    stop words from a set/list iterable. Uses ~gensim.utils.tokenize
    internally.
    """
    stop_words = stop_words if stop_words else SPACY_STOP_WORDS
    return [
        [tok for tok in gensim.utils.simple_preprocess(seq, deacc=deacc)
         if tok not in stop_words] for seq in doc
    ]
