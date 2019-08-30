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

    # TODO: FIX ISSUE WITH RETURNING NOT-FOUND INDEXES.
    '''Word pattern finder from a list of texts.
    * key[`loc`] -> tuple[s]:
        If multiple matches found: group (tuple)s, per index.
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


def test_all_indexes_with_matching_word():
    # easy test: assuming all indexes docs have the matching
    # word it should print the locations and a consistent id.
    matches = pattern_search('string', docs[0:4])
    for m in matches:
        print(m)


if __name__ == '__main__':
    test_all_indexes_with_matching_word()
