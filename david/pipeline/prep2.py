import re
import string
import unicodedata

import contractions
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pattern.text.en import tag

nltk_stop_words = nltk.corpus.stopwords.words('english')


def encode_ascii(text: str):
    ascii_s = unicodedata.normalize('NFKD', text)
    ascii_s = ascii_s.encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return ascii_s


def nltk_tokenizer(text: str):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def expand_contractions(text: str, slang=True):
    return contractions.fix(text, slang=slang)


def _penn_to_wn_tags(pos_tag: str):
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


def annotate_pos_tag(text: str):
    '''Annotates text tokens with pos tags, uses the wordnet
    part-of-speech class from nltk.
    '''
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), _penn_to_wn_tags(pos_tag))
                         for word, pos_tag in tagged_text]
    return tagged_lower_text


def lemmatize_text(text: str):
    # lemmataze text based on pos tags.
    wnl = WordNetLemmatizer()
    pos_tagged_text = annotate_pos_tag(text)
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


def remove_stopwords(text: str, stop_words: list = nltk_stop_words):
    tokens = nltk_tokenizer(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def remove_repeated_characters(tokens: list):
    '''Corrects repeating characters from tokens.

        >>> remove_repeated_characters(['finallllyyyy'])[0]
    'finally'
    '''
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def normalize_corpus(corpus: list, tokenize=False):
    normalized = []
    for text in corpus:
        text = encode_ascii(text)
        text = expand_contractions(text)
        text = lemmatize_text(text)
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized.append(text)
        if tokenize:
            text = nltk_tokenizer(text)
            normalized.append(text)
    return normalized
