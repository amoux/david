import collections
import random
import string
from typing import Dict, List, Optional, Tuple
from urllib.request import URLError, urlopen, Request

from bs4 import BeautifulSoup


def split_train_test(
        doc: List[str],
        n: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """Randomly split a doc into train and test iterables.

    Parameters:
    ----------

    `doc` (list[str]):
        A doc of iterable strings that will be split.

    `n` (Optional[int], default=None):
        The number of items in the doc to consider to subset.
        e.g, If you want to use 100 samples from 1000 samples then,
        < Train[80], Test[20] > will be the output.

    Returns (List[str], List[str]): Two iterables - train_doc is 8/10
        of the total while the test_doc is 2/10 of the total.

    """
    random.seed(12345)
    random.shuffle(doc)
    if not n or n > len(doc):
        n = len(doc)
    train_doc = doc[: int(0.8 * n)]
    test_doc = doc[int(0.8 * n): n]
    return train_doc, test_doc


def get_vocab_size(doc: List[str]) -> int:
    word_map = collections.Counter(doc)
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


def extract_text_from_url(url: str, headers: Dict[str, str] = None) -> str:
    """Extracts all text found in the webpage.

    Note: Returns an empty string if the connection to the URL failed.

    Usage:
        >>> url = "https://docs.python.org/3/faq/general.html"
        >>> extract_text_from_url(url)
        ...
        'Contents General Python FAQ General Information What is Python?...'

    """
    if not headers:
        headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        reqs = Request(url, headers=headers)
        page = urlopen(reqs)
        soup = BeautifulSoup(page, features="lxml")
        text = " ".join(map(lambda p: p.text, soup.find_all("p")))
    except URLError:
        return ""
    else:
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


def change_case(string: str) -> str:
    """Change case from camel case to snake case."""
    case = [string[0].lower()]
    for char in string[1:]:
        if char in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            case.append("_")
            case.append(char.lower())
        else:
            case.append(char)
    return "".join(case)
