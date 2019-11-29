import collections


def get_vocab_size(text: str):
    word_map = collections.Counter(text.split())
    unique_words = len(word_map.keys())
    vocab_size = int(unique_words)
    return vocab_size


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
