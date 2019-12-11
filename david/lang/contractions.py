import textsearch

from .en_lexicon import DAVID_CONTRACTIONS


def replace_contractions(sequence: str):
    """Uses the TextSearch library along with DAVID_CONTRACTIONS."""
    ts = textsearch.TextSearch(case="ignore", returns="norm")
    ts.add(DAVID_CONTRACTIONS)
    return ts.replace(sequence)
