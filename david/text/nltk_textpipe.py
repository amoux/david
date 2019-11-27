import re
import string
import unicodedata

import contractions
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pattern.text.en import tag as pattern_pos_tagger

from ..lang import NLTK_STOP_WORDS


def expand_contractions_basic(sentence: str, contraction_mapping: dict):
    contractions_pattern = re.compile(
        '({})'.format('|'.join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


def expand_contractions(text: str, leftovers=True, slang=True):
    return contractions.fix(text, leftovers=leftovers, slang=slang)


def encode_ascii(text: str):
    ascii_s = unicodedata.normalize('NFKD', text)
    ascii_s = ascii_s.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return ascii_s


def nltk_tokenizer(text: str):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def treebank_to_wordnet_postag(pos_tag: str):
    # convert penn treebank tag wordnet tag.
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def wordnet_postag_annotator(text: str):
    """Annotates text tokens with pos tags, uses the wordnet part-of-speech
    class from nltk."""
    tagged_text = pattern_pos_tagger(text)
    tagged_lower_text = [
        (word.lower(), treebank_to_wordnet_postag(pos_tag))
        for word, pos_tag in tagged_text
    ]
    return tagged_lower_text


def wordnet_lemmatizer(text: str):
    # lemmataze text based on pos tags.
    wnl = WordNetLemmatizer()
    pos_tagged_text = wordnet_postag_annotator(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text


def remove_special_characters(text: str):
    tokens = nltk_tokenizer(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(
        None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_stopwords(text: str, stop_words: list = None):
    if not stop_words:
        stop_words = NLTK_STOP_WORDS
    tokens = nltk_tokenizer(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_repeated_characters(tokens: list):
    """Corrects repeating characters from tokens (WordNet).

        >>> remove_repeated_characters(['finallllyyyy'])[0]
    'finally'
    """
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def preprocess_doc(doc: str,
                   contractions=True,
                   lemma=False,
                   special_chars=True,
                   stopwords=True,
                   tokenize=False):
    """NLTK text preprocessing for a string of text."""
    doc = encode_ascii(doc)
    if contractions:
        doc = expand_contractions(doc)
    if lemma:
        doc = wordnet_lemmatizer(doc)
    if special_chars:
        doc = remove_special_characters(doc)
    if stopwords:
        doc = remove_stopwords(doc)
    if tokenize:
        doc = nltk_tokenizer(doc)
    return doc


def preprocess_docs(docs: list,
                    contractions=True,
                    lemma=False,
                    special_chars=True,
                    stopwords=True,
                    tokenize=False):
    """NLTK text preprocessing for a list of texts."""
    normalized = []
    for doc in docs:
        normalized.append(preprocess_doc(
            doc=doc, contractions=contractions,
            lemma=lemma, special_chars=special_chars,
            stopwords=stopwords, tokenize=tokenize))
    return normalized
