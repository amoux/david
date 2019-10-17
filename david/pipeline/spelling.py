"""Spelling Corrector in Python 3; see http://norvig.com/spell-_correct.html

Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""

import collections
import re


class SpellCorrect(object):
    """Spelling Corrector by Peter Norvig."""

    DEFAULT_CORPUS = 'big.txt'
    WORD_COUNTS = None

    def __init__(self, corpus_path: str = None):
        """SpellCorrect uses word frquencies for matching the _correct
        grammar of a word.

        `corpus_path` : (str, default=None, 'big.txt')
            If the left as None, it imports `big.txt` file as
            the dataset for _correct spelled words.

        """
        self.corpus_path = corpus_path
        if not self.corpus_path:
            self.corpus_path = self.DEFAULT_CORPUS
        self.corpus = self.tokens(open(self.corpus_path).read())
        self.WORD_COUNTS = collections.Counter(self.corpus)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def most_common(self, top_n=10):
        return self.WORD_COUNTS.most_common(top_n)

    def tokens(self, text):
        """Get all the words from the corpus."""
        return re.findall('[a-z]+', text.lower())

    def correct_text(self, text):
        """Correct all the words within a text,
        returning the corrected text.
        """
        return re.sub('[a-zA-Z]+', self.correct_match, text)

    def correct_match(self, match):
        """Spell correct word in match, and preserve
        proper upper/lower/title case.
        """
        word = match.group()

        def case_of(text):
            """Return the case-function appropriate
            for text: upper, lower, title, or just str.
            """
            return (str.upper if text.isupper() else
                    str.lower if text.islower() else
                    str.title if text.istitle() else
                    str)
        return case_of(word)(self._correct(word.lower()))

    def _known(self, words):
        # Return the subset of words that are actually
        # in our WORD_COUNTS dictionary.
        return {w for w in words if w in self.WORD_COUNTS}

    def _edits0(self, word):
        # Return all strings that are zero edits away
        # from the input word (i.e., the word itself).
        return {word}

    def _edits1(self, word):
        # Return all strings that are one edit away from the input word.

        def splits(word):
            # Return a list of all possible (first, rest) pairs
            # that the input word is made of.
            return [(word[:i], word[i:]) for i in range(len(word)+1)]

        pairs = splits(word)
        deletes = [a+b[1:] for (a, b) in pairs if b]
        transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
        replaces = [a+c+b[1:] for (a, b) in pairs for c in self.alphabet if b]
        inserts = [a+c+b for (a, b) in pairs for c in self.alphabet]

        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        # Return all strings that are two edits away from the input word.
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

    def _correct(self, word):
        # Get the best _correct spelling for the input word.
        # Priority is for edit distance 0, then 1, then 2
        # else defaults to the input word itself.
        candidates = (self._known(self._edits0(word)) or
                      self._known(self._edits1(word)) or
                      self._known(self._edits2(word)) or
                      [word])
        return max(candidates, key=self.WORD_COUNTS.get)
