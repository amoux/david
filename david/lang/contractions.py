import textsearch

from .en_lexicon import DAVID_CONTRACTIONS


class TextSearchContractions:
    """Fix words from a large set of contractions (social-media-slang)."""
    EN_CONTRACTIONS = DAVID_CONTRACTIONS

    def __init__(self, case="ignore", returns="norm"):
        self.ts = textsearch.TextSearch(case=case, returns=returns)
        self.ts.add(self.EN_CONTRACTIONS)

    def fix(self, sequence: str):
        """Fix all the contractions found in a given sequence."""
        return self.ts.replace(sequence)

    def add(self, keyword: str, replacement: str):
        """Add a new keyword and replacement value to the contraction set."""
        self.ts.add(k=keyword, v=replacement)

    def add_many(self, mapping_dict: dict):
        """Update from a dictionary of many key/value pairs."""
        self.ts.add(mapping_dict)
