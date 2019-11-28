import string


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
            for token in tokens
        ]
        tokens = list(filter(lambda seq: len(seq) > min_seqlen, tokens))
        cleantokens.append(tokens)
    return cleantokens
