import collections
import logging
import os
import subprocess
import time
import urllib.request as urllib
import zipfile
from typing import Dict

from tqdm import tqdm

logger = logging.getLogger(__name__)


_BERT_MODEL_URLS = {
    "base-uncased": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "large-uncased": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
    "base-cased": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip",
    "large-cased": "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip",
    "base-multi-lang-new": "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
    "base-multi-lang-old": "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip",
}

BERT_MODELS = {
    "uncased-sm": [_BERT_MODEL_URLS["base-uncased"]],
    "uncased-lg": [_BERT_MODEL_URLS["large-uncased"]],
    "cased-sm": [_BERT_MODEL_URLS["base-cased"]],
    "cased-lg": [_BERT_MODEL_URLS["large-cased"]],
    "multi-lang-v2": [_BERT_MODEL_URLS["base-multi-lang-new"]],
    "multi-lang-v1": [_BERT_MODEL_URLS["base-multi-lang-old"]],
    "all": _BERT_MODEL_URLS.values(),
}


class TQDM(tqdm):
    def update_stream(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _extract_compressed_file(
    filename: str, save_path: str, extension: str = None
) -> None:
    """Extract a compressed file to directory.

    `filename`: Compressed file.
    `save_path`: Extract to directory.
    `extension`: Extension of the file; Otherwise, attempts to extract extension
        from the filename.
    """
    logger.info("Extracting {}".format(filename))
    if extension is None:
        basename = os.path.basename(filename)
        extension = basename.split(".", 1)[1]
    if "zip" in extension:
        with zipfile.ZipFile(filename, "r") as zip_:
            zip_.extractall(save_path)
    elif "tar.gz" in extension or "tgz" in extension:
        subprocess.call(["tar", "-C", save_path, "-zxvf", filename])
    elif "tar" in extension:
        subprocess.call(["tar", "-C", save_path, "-xvf", filename])
    logger.info("Extracted {}".format(filename))


def load_bert(
    model: str, save_path: str = "models", extension: str = ".zip"
) -> Dict[str, str]:
    """Download or load original BERT models.

    `model`: The name of the bert model to download. Choose one of the following:
        'uncased-sm', 'uncased-lg', 'cased-sm', 'cased-lg', 'multi-lang-v2',
        'multi-lang-v1'. If you need to download all models simply pass 'all'.
    `save_path`: The path directory where the models will be downloaded and unzipped.

    Usage:
        >>> bert_model_paths = load_bert("uncased-sm", save_path="models")
        >>> bert_model_paths.keys()
        dict_keys(['meta', 'index', 'config', 'vocab', 'data'])
        # Each key contains the absolute file path.
        >>> bert_model_paths["vocab"]
        '$HOME/../models/uncased_L-12_H-768_A-12/vocab.txt'
    """
    if model not in BERT_MODELS.keys():
        raise ValueError(
            "{}, is not a valid name. Choose one: {}".format(model, BERT_MODELS.keys())
        )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_file_paths = dict()

    for bert_url in BERT_MODELS[model]:
        filename = os.path.basename(bert_url)
        filepath = os.path.join(save_path, filename)
        if not os.path.isfile(filepath):
            logger.info("Directory for {} not found. Downloading...".format(model))
            # Start downloading the compressed file and display progress.
            with TQDM(
                unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
            ) as t:
                urllib.urlretrieve(
                    bert_url, filepath, reporthook=t.update_stream, data=None
                )
            # Extract the files from the compressed file downloaded.
            _extract_compressed_file(filepath, save_path, extension=extension)
        # Build the dictionary from the model's files. Assing clean key names.
        key_names = ["meta", "index", "config", "vocab", "data"]
        extlen = len(extension)
        model_file_paths = {
            k: os.path.abspath(os.path.join(filepath[:-extlen], f))
            for f, k in zip(os.listdir(filepath[:-extlen]), key_names)
        }
    return model_file_paths
