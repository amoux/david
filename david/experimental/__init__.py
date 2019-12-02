import math
import os


def norm_sequence_dist(samples: list,  max_seqlen: int, seqlen_std: float):
    """Method works only if the sequences have been tokenized to sentences.

    NOTE: the method `transform_texts` in this module is just an example
    of the preprocessing needed before passing a list of samples here.
    """
    def compute_cosine_split(n_samples, max_seqlen, seqlen_std):
        # compute the density distribution.
        max_p = round(max_seqlen % seqlen_std)
        cos_x = round(math.cos(max_p), 1)
        return int(n_samples * cos_x)

    cos_split = compute_cosine_split(len(samples), max_seqlen, seqlen_std)
    sort_seqs = sorted(samples, key=len)
    left_dist, right_dist = (
        len(sort_seqs[:cos_split]),
        len(sort_seqs[cos_split:])
    )
    norm_samples = sort_seqs[right_dist:left_dist]
    return norm_samples
