import re
from collections import Counter
from itertools import groupby

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


def remove_duplicated_words(text: str):
    return " ".join(Counter([word for word in text.split()]).keys())


def pattern_search(word: str, texts: list):
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
