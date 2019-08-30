import re
from itertools import groupby
from typing import Dict, Iterable, List, Sequence

docs = [
    'Scan through string looking for',
    'through string looking for the',
    'string looking string string string',
    'string looking for the first string',
    # DEBUG: this string doesnt have the word it duplicates the id.
    'Scan looking for the'
]


def pattern_search(word: Sequence[str],
                   texts: Iterable[str]) -> Iterable[Dict]:
    '''
    Word pattern finder from a list of texts.

    * key[`loc`] -> tuple[s]:
        If multiple matches found: group (tuple)s, per index.

    * Returns the following format:
    (NOTE: words not matched are not returned)

    >>> docs = ['string looking for the first string',]
    >>> for m in pattern_search('string', docs): print(m)

    {
        'id': 3,
        'loc': [(0, 6), (29, 35)],
        'endpos': 35,
        'match': 'string'
    }
    '''
    pattern = re.compile(word)
    temp_list = list()
    for idx, text in enumerate(texts):
        for token in pattern.finditer(text):
            matches = {
                'id': idx,
                'loc': [(loc.span()) for loc in pattern.finditer(text)],
                'endpos': token.endpos,
                'match': token.string[token.start():token.end()]
            }
        temp_list.append(matches)
    return [i[0] for i in groupby(temp_list)]
