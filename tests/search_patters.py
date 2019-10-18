import re
from collections import Counter
from itertools import groupby
from typing import Dict, Iterable, List, Sequence

PATTERN_SEARCH_DATA = [
    'Scan through string looking for',
    'through string looking for the',
    'string looking string string string',
    'string looking for the first string',
    'Scan looking for the'
]

DUPLICATE_DATA = [
    'Python Python Python Python is great and Java is not so great',
    'Python Python Python Python is awesome',
    'Python is a language close to human language.',
    'PythonPythonPython. Python:, yep said it three times',
    'Python Python Python Python Python Python Python Python',
    'Python Python programming! and yeah Python Python.'
]


def remove_duplicates(text):
    """Removes duplicate words found in a sequence of words.
    * Split text string separated by space.
    * Joins two adjacent elements in iterable way.
    * Uses the Counter method have strings as key and
    their frequencies as value.
    * Returns a joined adjacent of elements.

    >>> one_text = 'Python Python Python is awesome! Python'
    >>> remove_duplicates(one_text)
    ...
    'Python is awesome!'
    """
    text = text.split(' ')
    for idx in range(0, len(text)):
        text[idx] = ''.join(text[idx])
    unique = Counter(text)
    return ' '.join(unique.keys())


def pattern_search(
        word: Sequence[str],
        texts: Iterable[str],
) -> Iterable[Dict]:
    """Word pattern finder from a list of texts.

    Returns the following format, words not matched are not returned.

    key['loc'] : tuple[s]:
        If multiple matches found: group (tuple)s, per index.

    >>> docs = ['string looking for the first string',]
    >>> for m in pattern_search('string', docs): print(m)
        {'id': 3,
        'loc': [(0, 6), (29, 35)],
        'endpos': 35,
        'match': 'string'}
    """
    pattern = re.compile(word)
    temp_list = list()
    for idx, text in enumerate(texts):
        for token in pattern.finditer(text):
            matches = {
                'id': idx,
                'loc': [(loc.span()) for loc in pattern.finditer(text)],
                'endpos': token.endpos,
                'match': token.string[token.start():token.end()]}
        temp_list.append(matches)
    return [i[0] for i in groupby(temp_list)]
