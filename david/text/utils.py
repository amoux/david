import collections
import random
import string
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, URLError, urlopen

import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from wasabi import msg
from wordcloud import WordCloud

from ..lang import SPACY_STOP_WORDS


def largest_sequence(sequences: Iterable[Sequence[List[int]]]) -> int:
    """Obtain the value of the largest sequence from an iterable.

    Usage:
        >>> largest_sequence([[ 32, 4, 45], [1, 9], [2]])
            3
    """
    largest = []
    for seq in sequences:
        try:
            largest.append(len(seq))
        except TypeError or ValueError:
            msg.fail(
                "Sequences must be a list of iterables, found "
                f"item < {str(seq)} > : {type(seq)} not iterable."
            )
            break
    return np.max(largest)


def split_train_test(
    document: List[str], n: Optional[int] = None, seed=12345, subset=0.8
) -> Tuple[List[str], List[str]]:
    """Randomly split a document into train and test iterables.

    `document`: An iterable strings that will be split.
    `n`: The number of items in the doc to consider to subset.
        e.g, If you want to use 100 samples from 1000 samples then,
        < Train[80], Test[20] > will be the output.

    Returns (List[str], List[str]): Two iterables - train_document is 8/10
        of the total while the test_document is 2/10 of the total.
    """
    random.seed(seed)
    random.shuffle(document)
    if not n or n > len(document):
        n = len(document)
    train_doc = document[: int(subset * n)]
    test_doc = document[int(subset * n) : n]
    return train_doc, test_doc


def get_vocab_size(document: List[str]) -> int:
    """Obtain the vocab size from a document of strings based on unique words."""
    word_count = collections.Counter(document)
    vocab_size = int(len(word_count.keys()))
    return vocab_size


def is_tokenized_doc(obj):
    """Check whether the object is an iterable of sequence tokens.

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
    """Extract all texts found in the webpage.

    Note: Returns an empty string if the connection to the URL failed.

    Usage:
        >>> url = "https://docs.python.org/3/faq/general.html"
        >>> extract_text_from_url(url)
        ...
        'Contents General Python FAQ General Information What is Python?...'

    """
    if not headers:
        headers = {"User-Agent": "Mozilla/5.0"}
    try:
        reqs = Request(url, headers=headers)
        page = urlopen(reqs)
        soup = BeautifulSoup(page, features="lxml")
        text = " ".join(map(lambda p: p.text, soup.find_all("p")))
    except URLError:
        return ""
    else:
        return text


def clean_tokens(
    document: Iterable[List[Sequence[str]]], discard_punct="_", min_seqlen=1
):
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
    for tokens in document:
        tokens = [
            "".join([seq for seq in token if seq not in punctuation])
            for token in tokens
        ]
        tokens = list(filter(lambda seq: len(seq) > min_seqlen, tokens))
        cleantokens.append(tokens)
    return cleantokens


def complete_sentences(document: List[str]) -> List[str]:
    """Add a period at the end of a complete sentence.

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
    for sent in document:
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


def build_wordcloud(
    doc: list,
    img_name: str = "wordcloud",
    width: int = 1600,
    height: int = 600,
    margin=3,
    max_words: int = 200,
    max_font_size=150,
    image_dpi=900,
    random_state=62,
    background_color="black",
    stop_words: list = None,
):
    """Build a word cloud image from text sequences."""
    if not stop_words:
        stop_words = SPACY_STOP_WORDS

    wordcloud = WordCloud(
        width=width,
        height=height,
        margin=margin,
        max_words=max_words,
        max_font_size=max_font_size,
        random_state=random_state,
        background_color=background_color,
        stopwords=stop_words,
    ).generate(str(doc))
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    fig.savefig(img_name, dpi=image_dpi)
