from __future__ import unicode_literals

import collections
import re
import string
import unicodedata
from string import printable as PRINTABLE
from typing import Generator, List, Tuple

import emoji
import gensim
import nltk
import spacy
import textblob
from emoji.unicode_codes import UNICODE_EMOJI

from ..lang import SPACY_STOP_WORDS, replace_contractions


def unicode_to_ascii(sequence: str) -> str:
    """Convert the unicode to ASCII."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", sequence)
        if unicodedata.category(char) != "Mn"
    )


def remove_urls(string: str) -> str:
    """Remove any url link from a string sequence."""
    pattern = re.compile(r"(http\S+)")
    return pattern.sub(" ", string)


def string_printable(string: str) -> str:
    """Return a string of ASCII characters which are consired printable.

    This method checks whether the characters are legal in combination of:
        digits, ascii_letters, punctuation, whitespace and unicode_emoji.
    """
    return "".join(
        [char for char in string if (char in PRINTABLE or char in UNICODE_EMOJI)]
    )


def clean_tokenization(sequence: str) -> str:
    """Clean up spaces before punctuations and abreviated forms."""
    return (
        sequence.replace(" .", ".")
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


def extract_emojis(sequence: str):
    """Extract all emoji characters found in a sequence."""
    return " ".join([char for char in sequence if char in emoji.UNICODE_EMOJI])


def get_sentiment_polarity(sequence: str):
    """NOTE: Replacing all Textblob methods with optimized sentiment models."""
    return textblob.TextBlob(sequence).sentiment.polarity


def get_sentiment_subjectivity(sequence: str):
    """NOTE: Replacing all Textblob methods with optimized sentiment models."""
    return textblob.TextBlob(sequence).sentiment.subjectivity


def lemmatizer(doc: list):
    """Nltk wordnet lemmmatizer helper function."""
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
        lemmas.append([tok.lemma_ for tok in sequence if tok.pos_ in postags])
    return lemmas


def spacy_sentence_tokenizer(doc: List[str]) -> List[str]:
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


def treebank_to_wordnet_pos(pos_tag: str) -> str:
    """NLTK POS-tagger, converts penn treebank tags wordnet tags."""
    if pos_tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    elif pos_tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    elif pos_tag.startswith("N"):
        return nltk.corpus.wordnet.NOUN
    elif pos_tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    else:
        return None


def part_of_speech_annotator(sequence: str) -> List[Tuple[str, str]]:
    """Annotates text tokens with pos tags, uses NLTK's Wordnet POS."""

    # This is a temporary patch for eliminating the need for the pattern
    # library. I will be using spacy's pos tagging for now but this method
    # of lemmatizing text is not efficient (I will work on a better solution).
    # In the meantime this will work for apps that are using these methods.

    nlp = spacy.load("en_core_web_sm")
    sequence = nlp(sequence)
    return [(token.text, treebank_to_wordnet_pos(token.pos_)) for token in sequence]


def part_of_speech_lemmatizer(sequence: str):
    """Lemmatize sequences based on part-of-speech tags."""
    Lemmatizer = nltk.stem.WordNetLemmatizer()
    tagged = part_of_speech_annotator(sequence)
    return " ".join(
        [Lemmatizer.lemmatize(word, tag) if tag else word for word, tag in tagged]
    )


def normalize_whitespace(sequence: str):
    """Normilize excessive whitespace from a string without formating the original form."""
    LINEBREAK = re.compile(r"(\r\n|[\n\v])+")
    NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)
    return NONBREAKING_SPACE.sub(" ", LINEBREAK.sub(r"\n", sequence)).strip()


def normalize_wiggles(sequence, min_words=10):
    """Normalize wiggles from sequences.

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
    return " ".join(collections.Counter([tok for tok in sequence.split()]).keys())


def remove_repeating_characters(sequence: str):
    """Remove repeating characters from one/multiple words in a sequence."""

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
    """Remove punctuation from a string sequence.
    
    The method uses contractions to correctly remove punctuation without
    hurting the meaning of words like `"y'all"` -> `"you are all"` should
    be done before actually removing all punctuation.
    
    NOTE: punctuations "-" and "_" are not removed.
    """
    punctuation = re.escape(string.punctuation)
    punctuation = re.compile("[{}]".format(punctuation))
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


def preprocess_sequence(
    sequence: str,
    contractions=True,
    lemmatize=False,
    punctuation=True,
    norm_chars=True,
    rm_stopwords=True,
    stop_words=None,
    tokenize=False,
):
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


def preprocess_doc(
    doc: list,
    contractions=True,
    lemmatize=False,
    punctuation=True,
    norm_chars=True,
    rm_stopwords=True,
    stop_words=None,
    tokenize=False,
):
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
        [
            tok
            for tok in gensim.utils.simple_preprocess(seq, deacc=deacc)
            if tok not in stop_words
        ]
        for seq in doc
    ]
