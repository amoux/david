import math
import os

import numpy as np


def norm_sequence_dist(samples: list, max_seqlen: int, seqlen_std: float):
    """Normalize the distribution density spread from an iterable document.

    NOTE: Method works only if the sequences have been tokenized to sentences.
    """

    def compute_cosine_split(n_samples, max_seqlen, seqlen_std):
        # compute the density distribution.
        max_p = round(max_seqlen % seqlen_std)
        cos_x = round(math.cos(max_p), 1)
        return int(n_samples * cos_x)

    cos_split = compute_cosine_split(len(samples), max_seqlen, seqlen_std)
    sort_seqs = sorted(samples, key=len)
    left_dist, right_dist = (len(sort_seqs[:cos_split]), len(sort_seqs[cos_split:]))
    norm_samples = sort_seqs[right_dist:left_dist]
    return norm_samples


def similarity(token: str, k=5, norm=None, vocab_matrix=None, tokenizer=None):
    """Return the most similar tokens based on one single token string (Experimental).
    
    The following norms can be calculated:
    ```bash
    ===== ==========================
    ord   norm for vectors
    ===== ==========================
    None   2-norm
    inf    max(abs(x))
    -inf   min(abs(x))
    0      sum(x != 0)
    other  sum(abs(x)**ord)**(1./ord)
    =====  ============================
    ```
    Usage:
    ```python
    from david.tokenizers import Tokenizer
    from david.models import GloVe
    tokenizer = Tokenizer(document=dataset)
    tokenizer.vocabulary_to_frequency(mincount=1)
    matrix = GloVe.fit_embeddings(tokenizer.vocab_index, vocab_dim="100d")
    # query top-k most similar tokens given a single token.
    similarity('youtube', k=5, norm=None, vocab_matrix=matrix, tokenizer=tokenizer)
    ...
    [('twitter', 0.7946596437148128),
     ('videos', 0.7294430511868724),
     ('video', 0.7010882708151076),
     ('web', 0.6723791043550629)]
    ```
    """
    # division by zero error message patch.
    np.seterr(divide="ignore", invalid="ignore")

    embedding = vocab_matrix[tokenizer.vocab_index[token]]
    if norm == "max":
        norm = max(abs(embedding))
    if norm == "min":
        norm = min(abs(embedding))

    dst = (
        np.dot(vocab_matrix, embedding)
        / np.linalg.norm(vocab_matrix, ord=norm, axis=1)
        / np.linalg.norm(embedding)
    )

    token_ids = np.argsort(-dst)
    id2tok = {idx: tok for tok, idx in tokenizer.vocab_index.items()}
    return [(id2tok[i], dst[i]) for i in token_ids if i in id2tok][1 : k + 1]
