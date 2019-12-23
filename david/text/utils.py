import collections
import string
from typing import List
from urllib.request import urlopen

from bs4 import BeautifulSoup


def get_vocab_size(text: str):
    word_map = collections.Counter(text.split())
    unique_words = len(word_map.keys())
    vocab_size = int(unique_words)
    return vocab_size


def is_tokenized_doc(obj):
    """Checks whether the object is an iterable of sequence tokens.

    Minimum valid size of a tokenized document e.g,. `[["hey"]]`.
    """
    if (
        isinstance(obj, list)
        and len(obj) > 0
        and isinstance(obj[0], list)
        and isinstance(obj[0][0], str)
    ):
        return True
    else:
        return False


def extract_text_from_url(url: str) -> str:
    """Extracts all text found in the webpage.

    Usage:
        >>> extract_text_from_url("https://docs.python.org/3/faq/general.html")
        'Contents General Python FAQ General Information What is Python?...'

    """
    page = urlopen(url)
    soup = BeautifulSoup(page, features="lxml")
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
    return text


def clean_tokens(doc: list, discard_punct="_", min_seqlen=1):
    """Remove tokens consisting of punctuation and/or by minimum N sequences.

    Usage:
        >>> clean_tokens(
                [['x', 'Hello!', 'keep', 'this_punct', '#2020'],
                 ['H', '', 'tokens', 'b***',  '[::[hidden]', '/,']])
        ...
        '[['Hello', 'keep', 'this_punct', '2020'], ['tokens', 'hidden']]'
    """
    # discarding punctuation can be further extended.
    punctuation = set([p for p in string.punctuation])
    punctuation.discard(discard_punct)
    cleantokens = list()
    for tokens in doc:
        tokens = [
            ''.join([seq for seq in token if seq not in punctuation])
            for token in tokens]
        tokens = list(filter(lambda seq: len(seq) > min_seqlen, tokens))
        cleantokens.append(tokens)
    return cleantokens


def complete_sentences(raw_doc: List[str]) -> List[str]:
    """Adds a period at the end of a complete sentence.

    NOTE: A sentence is recognized if the first word starts with an
    uppercase letter while extending the next items into itself and
    then adding a period before the next upper case letter is found.

    Usage: Example of how the sentences should be aligned.

        >>> document = ['hello world',
                         'I am happy',
                         'when it works',
                         'My name is X',
                         'and I program',
                         'with python',
                         'This is a',
                         'sentence too']
        ...
        >>> complete_sentences(document)
        ['hello world.',
         'I am happy when it works.',
         'My name is X and I program with python.',
         'This is a sentence too.']

    TODO: Extend conditions based on POS tags and not just rely on upper case
    letters.

    """
    doc = list()
    for sent in raw_doc:
        doc.append(sent if not sent.split()[0].istitle() else f"<sos>{sent}")
    doc = " ".join(doc).split("<sos>")
    doc = [f"{sent.strip()}." for sent in doc]
    return doc
