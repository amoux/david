"""
Glove embeddings module.

-----------------------
The `Glove` module needs a bit of cleaning to do when
loading the files. I have been working hard to have a
complete set of modules for all my word embedding needs!
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np
from wasabi import msg


def _load_vocab_files(glove_dirname: str) -> Dict[str, str]:
    GLOVE_DIR = os.environ.get("GLOVE_DIR")
    GLOVE6B_PATH = os.path.join(GLOVE_DIR, glove_dirname)
    GLOVE6B_FILES = os.listdir(GLOVE6B_PATH)
    glove_files = {}
    for i, file in enumerate(GLOVE6B_FILES):
        ndim = [d for d in file.split(".") if d.endswith("d")][: i + 1][0]
        glove_files[ndim] = os.path.join(GLOVE6B_PATH, file)
    return glove_files


class GloVe:
    """Global Vectors for Word Representation.

    Returns an embedded matrix containing the pretrained word embeddings.
    """

    vocab_files = _load_vocab_files("glove.6B")

    @staticmethod
    def build_vocabulary(vocab_file: str) -> Dict[str, np.float]:
        """Build glove embeddings from file.

        Creates a dictionary with words as keys and the corresponding
        `N-dim` vectors as values, in the form of an array.
        """
        vocab_path = Path(vocab_file)
        if not vocab_path.exists():
            msg.fail(f"Could not find glove file in {vocab_file}")
        else:
            msg.good(f"Glove embeddings loaded from path: {vocab_file}")

        with vocab_path.open("r", encoding="utf8") as file:
            glove_embeddings = {}
            for line in file:
                values = line.split()
                token = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                glove_embeddings[token] = vector

            return glove_embeddings

    @staticmethod
    def load_vocabulary(vocab_ndim: str = "100d") -> Dict[str, np.float]:
        """Return the built GloVe embeddings given a available dimension."""
        vocab_file = GloVe.vocab_files[vocab_ndim]
        return GloVe.build_vocabulary(vocab_file)

    @staticmethod
    def fit_embeddings(
        vocab_index: Dict[str, int],
        vocab_dim: str = "100d",
        vocab_size: int = None,
        max_tokens: int = None,
    ):
        """Fit an indexed vocab with GloVe's pretrained word embeddings.

        vocab_index: A mapped index to tokens dictionary.
        vocab_dim: The vocab dimension to load from file.
        vocab_size: The size of the indexed vocabulary. If None,
            the vocab size will be calculated from the vocab index dict.
        max_tokens: Maximum number of tokens to consider embedding.
        """
        if max_tokens is None:
            max_tokens = len(vocab_index)
        if vocab_size is None:
            vocab_size = 1 + max_tokens
        dimensions = int(vocab_dim.replace("d", ""))

        msg.good(f"<✔(tokens={max_tokens}, ndim={dimensions}, vocab={vocab_size})>")
        msg.good("*** embedding vocabulary 👻 ***")

        glove_embeddings = GloVe.load_vocabulary(vocab_dim)
        vocab_embeddings = np.zeros((vocab_size, dimensions))
        for token, token_id in vocab_index.items():
            if token_id < max_tokens:
                try:
                    embedded_token = glove_embeddings[token]
                # skip any token not in glove's vocabulary
                except KeyError:
                    continue
                # otherwise, add the token_id and its embedding form
                else:
                    vocab_embeddings[token_id] = embedded_token

        return vocab_embeddings
