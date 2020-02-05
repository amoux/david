"""
Glove embeddings module.

The `Glove` module needs a bit of cleaning to do when
loading the files. I have been working hard to have a
complete set of modules for all my word embedding needs!
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np


class GloVe:
    """Global Vectors for Word Representation.

    Returns an embedded matrix containing the pretrained
    word embeddings.
    """

    GLOVE_DIR = os.environ.get("GLOVE_DIR")
    GLOVE6B_PATH = os.path.join(GLOVE_DIR, "glove.6B")
    GLOVE6B_FILES = os.listdir(GLOVE6B_PATH)

    vocab_files: Dict[str, str] = {}
    for i, glove_file in enumerate(GLOVE6B_FILES):
        ndim = [d for d in glove_file.split(".") if d.endswith("d")][: i + 1][0]
        vocab_files[ndim] = os.path.join(GLOVE6B_PATH, glove_file)

    def build_vocabulary(vocab_file: str) -> Dict[str, np.float]:
        """Load glove embeddings from file.
    
        Creates a dictionary with words as keys and the corresponding
        `N-dim` vectors as values, in the form of an array.
        """
        vocab_path = Path(vocab_file)
        if not vocab_path.exists():
            raise FileNotFoundError(f"Could't find glove file in {vocab_file}")

        with vocab_path.open("r", encoding="utf8") as glove_file:
            embeddings = {}
            for line in glove_file:
                ndims = line.split()
                token = ndims[0]
                embeddings[token] = np.asarray(ndims[1:], dtype="float32")
            return embeddings

    @staticmethod
    def fit_embeddings(
        vocab_index: Dict[str, int], vocab_dim="100d", vocab_size: int = None
    ):
        """Fit an indexed vocab with GloVe's pretrained word embeddings.

        `vocab_index`: A mapped index to tokens dictionary.
        `vocab_dim`: The vocab dimension to load from file.
        `vocab_size`: The size of the indexed vocabulary. If None,
        the vocab size will be calculated from the vocab index dict.
        """
        vocab_file = GloVe.vocab_files[vocab_dim]
        print(f"Loading vocab file from {vocab_file}")

        num_dim = int(vocab_dim.replace("d", ""))
        if vocab_index and vocab_size is None:
            vocab_size = 1 + len(vocab_index.keys())

        print(
            f"num-dim:({num_dim}), vocab-size: {vocab_size}",
            "\n*** embedding vocabulary...\n",
        )

        glove_embeddings = GloVe.build_vocabulary(vocab_file)
        vocab_embeddings = np.zeros((vocab_size, num_dim))
        for token, index in vocab_index.items():
            embedding = glove_embeddings.get(token)
            if embedding is not None:
                vocab_embeddings[index] = embedding

        return vocab_embeddings
