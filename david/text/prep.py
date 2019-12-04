from __future__ import unicode_literals

import collections
import re
import string
import unicodedata

import emoji
import nltk
import pattern
import spacy
import textblob
from nltk.tokenize.casual import (EMOTICON_RE, HANG_RE, WORD_RE,
                                  _replace_html_entities, reduce_lengthening,
                                  remove_handles)

from ..lang import SPACY_STOP_WORDS, TextSearchContractions


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


def encode_ascii(sequence: str):
    ascii_seq = unicodedata.normalize('NFKD', sequence)
    return ascii_seq.encode('ascii', 'ignore').decode('utf-8', 'ignore')


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


def spacy_sentence_tokenizer(doc: list, nlp_model='en_core_web_lg'):
    """Spacy sentence tokenizer.

    Args:
        docs (list): An iterable list containing texts.
        nlp_model (object): A spacy language model instance to use
        with the sentence tokenizer.
    """
    nlp = spacy.load(nlp_model)
    sentences = list()
    for line in doc:
        doc = nlp(line)
        for sent in doc.sents:
            sentences.append(sent.text)
    return sentences


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
    """Lemmataze sequences based on part-of-speech tags."""
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


def remove_punctuation(sequence: str):
    pattern = re.compile("[{}]".format(re.escape(string.punctuation)))
    tokens = nltk_word_tokenizer(sequence)
    return " ".join(filter(None, [pattern.sub("", tok) for tok in tokens]))


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
    sequence = normalize_whitespace(encode_ascii(sequence))
    if contractions:
        sequence = TextSearchContractions().fix(sequence)
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

    normalized = list()
    for sequence in doc:
        normalized.append(
            preprocess_sequence(
                sequence=sequence,
                contractions=contractions,
                lemmatize=lemmatize,
                punctuation=punctuation,
                norm_chars=norm_chars,
                rm_stopwords=rm_stopwords,
                stop_words=stop_words,
                tokenize=tokenize,
            )
        )
    return normalized
