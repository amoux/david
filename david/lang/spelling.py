"""Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html.

Copyright (c) 2007-2016 Peter Norvig MIT license:
www.opensource.org/licenses/mit-license.php
"""

import re
from collections import Counter
from typing import Dict, List, Pattern


class Speller:
    """Spell correction based on Peter Norvig's implementation."""

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    def __init__(
        self,
        filepath: str = None,
        document: List[str] = None,
        word_count: Dict[str, int] = None,
    ):
        """Speller uses word counts as a metric for correcting words.

        `filepath` : A file containing lines of string sequences.
        `document` : An iterable document of string sequences.
        `word_count` : An instance of `collections.Counter` with
            existing word count pairs.
        """
        self.word_count = word_count
        if filepath is not None:
            self.word_count_from_file(filepath)
        elif document is not None:
            self.word_count_from_doc(document)

    def word_count_from_file(self, filepath: str):
        """Load and tokenize texts into word count pairs from a file."""
        tokens = self.tokenize(open(filepath).read())
        self.word_count = Counter(tokens)

    def word_count_from_doc(self, document: List[str]):
        """Set the word count dictionary from a document of string sequences."""
        tokens = []
        for doc in document:
            tokens.extend(self.tokenize(doc))
        self.word_count = Counter(tokens)

    def most_common(self, k=10):
        """Return the most common words from the dictionary counter."""
        return self.word_count.most_common(k)

    def tokenize(self, sequence: str):
        """Regex based word tokenizer."""
        return re.findall("[a-z]+", sequence.lower())

    def correct_string(self, sequence: str):
        """Return the correct spell form a string sequence."""
        return re.sub("[a-zA-Z]+", self.correct_match, sequence)

    def correct_match(self, match: Pattern[str]):
        """Spell correct word in match, and preserve proper case."""
        word = match.group()

        def case_of(text):
            """Return the case-function appropriate for text."""
            return (
                str.upper
                if text.isupper()
                else str.lower
                if text.islower()
                else str.title
                if text.istitle()
                else str
            )

        return case_of(word)(self._correct(word.lower()))

    def _known(self, words):
        return {w for w in words if w in self.word_count}

    def _edits0(self, word):
        return {word}

    def _edits1(self, word):
        def splits(word):
            return [(word[:i], word[i:]) for i in range(len(word) + 1)]

        pairs = splits(word)
        deletes = [a + b[1:] for (a, b) in pairs if b]
        transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
        replaces = [a + c + b[1:] for (a, b) in pairs for c in self.alphabet if b]
        inserts = [a + c + b for (a, b) in pairs for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def _edits2(self, word):
        return {e2 for e1 in self._edits1(word) for e2 in self._edits1(e1)}

    def _correct(self, word):
        candidates = (
            self._known(self._edits0(word))
            or self._known(self._edits1(word))
            or self._known(self._edits2(word))
            or [word]
        )
        return max(candidates, key=self.word_count.get)
