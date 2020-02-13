from collections import Counter
from typing import Any, List, Tuple

from ..text.preprocessing import (normalize_whitespace, normalize_wiggles,
                                  remove_repeating_characters,
                                  sentence_tokenizer)


def split_youtube_audiences(
        db_batch: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Split Youtube audiences from a batch of comments.

    Usage:
        >>> db = CommentsSql("v2")
        >>> batch = db.fetch_comments("%how to code%")
        >>> replies, no_replies = split_youtube_audiences(batch)
        >>> print(len(replies), len(no_replies))
        >>> (8, 43)

    Returns two Iterables (comments with-replies, comments without-replies).
        Note: The attributes from the batch splits keep the attributes.

    """
    is_reply = list(filter(lambda i: len(i.cid) > 26, db_batch))
    no_reply = list(filter(lambda i: len(i.cid) == 26, db_batch))
    return (is_reply, no_reply)
