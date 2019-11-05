import math

from . import tensor_w2v


def compute_cosine_split(n_samples: int, max_seqlen: int, seqlen_std: float):
    max_p = round(max_seqlen % seqlen_std)
    cos_x = round(math.cos(max_p), 1)
    return int(n_samples * cos_x)


def norm_sequence_dist(samples: list,  max_seqlen: int, seqlen_std: float):
    # compute the split decision and get the cosine value.
    cos_split = compute_cosine_split(len(samples), max_seqlen, seqlen_std)
    sort_seqs = sorted(samples, key=len)
    left_dist, right_dist = (
        len(sort_seqs[:cos_split]),
        len(sort_seqs[cos_split:])
    )
    # flip outter distributions to normalization.
    norm_samples = sort_seqs[right_dist:left_dist]
    return norm_samples
