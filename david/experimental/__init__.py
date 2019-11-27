import collections
import math
import os
import string
import urllib
import zipfile

from ..text import encode_ascii, sent_tokenizer
from . import tensor_w2v


def transform_texts(texts):
    encoded_texts = list()
    for text in texts:
        encoded_texts.append(encode_ascii(text))
    return sent_tokenizer(encoded_texts)


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


def download_movielens_datasets(datasets,
                                savepath='datasets',
                                file_ext='.zip'):
    """Downloads zip files from urls and loads them in the right format.

    NOTE: This method can be improved to work with many datasets with
    different file extensions. And not just `.zip` files. key areas:

    feature_one : `filename = dataset[dataset.rfind('ml'):]`
        str.rfind() returns the start idx value of a file from a string (url)
        which I used to slice off the file extension: `file.ext` -> `file`.
    """
    extlen = len(file_ext)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    dataset_fpaths = collections.defaultdict(list)
    for dataset in datasets:
        filename = dataset[dataset.rfind('ml'):]
        filepath = os.path.join(savepath, filename)
        if not os.path.exists(filepath[:-extlen]):
            download = urllib.request.urlretrieve(dataset, filepath)
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(savepath)
        dataset_fpaths[filename[:-extlen]] = {
            fn[:-extlen]: os.path.join(filepath[:-extlen], fn)
            for fn in os.listdir(filepath[:-extlen])}
    return dataset_fpaths
