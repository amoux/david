"""
Glove embeddings module.

-----------------------
The `Glove` module needs a bit of cleaning to do when
loading the files. I have been working hard to have a
complete set of modules for all my word embedding needs!
"""

import os
from os import environ
from pathlib import Path
from typing import Dict

import numpy as np
from wasabi import msg


def load_glove_files(dirname: str) -> Dict[str, str]:
    # - load all glove files found within the directory
    # - return a dict with keys as the dimension name and
    # values as the location of the file {ndim: file_path}
    glove_dir = Path(environ.get("GLOVE_DIR"))
    if not glove_dir.exists():
        glove_dir.mkdir(parents=True)
    glove_dir = glove_dir.joinpath(dirname)

    glove_files = {}
    for i, file in enumerate(os.listdir(glove_dir)):
        ndim = [f for f in file.split(".") if f.endswith("d")][: i + 1][0]
        glove_files[ndim] = os.path.join(glove_dir, file)

    return glove_files


class GloVe:
    """Global Vectors for Word Representation.

    Returns an embedded matrix containing the pretrained word embeddings.
    """

    vocab_files = None
    glove_modules = {
        "6b": "glove.6B",
        "27b": "glove.twitter.27B",
        "34b": "glove.34B",
        "840b": "glove.840B",
    }

    def __init__(self, module: str = None):
        """Initialize a GloVe module.

        `module`: name of the modules available: `6b`, `27b`, `34b`, `840b`.
        """
        if module not in self.glove_modules:
            raise ValueError(
                f"Invalid module name, got '{module}'. "
                f"Valid names: {self.glove_modules.keys()}"
            )
        module_name = self.glove_modules[module]
        self.vocab_files = load_glove_files(module_name)

    @staticmethod
    def build(vocab_file: str) -> Dict[str, np.float]:
        """Build glove embeddings from file.

        Creates a dictionary with words as keys and the corresponding
        `N-dim` vectors as values, in the form of an array.
        """
        vocab_path = Path(vocab_file)
        if not vocab_path.exists():
            msg.fail(f" Could not find glove file in {vocab_file}")
        else:
            msg.good(f" Glove embeddings loaded from path: {vocab_file}")

        with vocab_path.open("r", encoding="utf8") as file:
            glove_embeddings = {}
            for line in file:
                values = line.split()
                token = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                glove_embeddings[token] = vector
            return glove_embeddings

    def load(self, ndim: str) -> Dict[str, np.float]:
        """Return the built GloVe embeddings given a available dimension."""
        vocab_file = self.glove_modules[ndim]
        return self.build(vocab_file)

    def embedd(self, ndim: str, vocab: Dict[str, int], vocab_size: int = None):
        """Fit an indexed vocab with GloVe's pretrained word embeddings.

        ndim: The vocab dimension to load from file.
        vocab: A mapped index to tokens dictionary.
        vocab_size: The size of the indexed vocabulary. If None,
            the vocab size will be calculated from the vocab index dict.
        max_tokens: Maximum number of tokens to consider embedding.
        """
        if vocab_size is None:
            vocab_size = len(vocab) + 1
        dimension = int(ndim.replace("d", ""))

        msg.good(f" < (dim={dimension}, vocab_size={vocab_size}) >")
        msg.good(" embedding vocabulary 👻")

        embedding_index = self.load(ndim)
        embedding_matrix = np.zeros((vocab_size, dimension))
        for token, index in vocab.items():
            embedding_vector = embedding_index.get(token)
            if embedding_vector is not None:
                # tokens not found in embedding index will be all-zeros
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    def __call__(self, ndim: str, vocab: Dict[str, int], size: int = None):
        """Fit an indexed vocab with GloVe's pretrained word embeddings."""
        return self.embedd(ndim, vocab, vocab_size=size)

    def __repr__(self):
        return f"< GloVe(dims={list(self.vocab_files.keys())}) >"
