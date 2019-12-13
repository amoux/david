from collections import Counter
from typing import Any, List, Tuple

from ..text.prep import (normalize_whitespace, normalize_wiggles,
                         remove_repeating_characters, sentence_tokenizer)


def split_youtube_audiences(
        db_batch: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Splits Youtube audiences from a batch of comments.

    Usage:
        >>> db = CommentsSQL("v2")
        >>> batch = db.search_comments("%how to code%")
        >>> replies, no_replies = split_youtube_audiences(batch)
        >>> print(len(replies), len(no_replies))
        >>> (8, 43)

    Returns two Iterables (comments with-replies, comments without-replies).
        Note: The attributes from the batch splits keep the attributes.

    """
    is_reply = list(filter(lambda i: len(i.cid) > 26, db_batch))
    no_reply = list(filter(lambda i: len(i.cid) == 26, db_batch))
    return (is_reply, no_reply)


def simple_database_preprocess(db_batch: List[Any]) -> List[str]:
    """Simple text preprocessing pipeline for CommentsSQL instace batches.

    Normalizes whitespaces, removes repeated characters/words, removes empty
    lines, filters out repeated strings (comments) and transforms all texts to
    sentences using spacy's sentence tokenizer.


    NOTE: This method should only be used for batches from the CommentSQL
    class and it removes the attributes returning only an Iterable list of
    text sequences.
    """
    sents = list(
        sentence_tokenizer(
            Counter(
                [
                    normalize_wiggles(
                        remove_repeating_characters(
                            normalize_whitespace(seq.text)
                        )
                    )
                    for seq in db_batch if seq != ""
                ]
            ).keys()
        )
    )
    return sents
