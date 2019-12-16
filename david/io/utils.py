import fnmatch
import io
import json
import logging
import os
import urllib
from typing import IO, Any, Generator, Iterable, List, Text, TextIO

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def walkdir(folder: str) -> Generator:
    """Walk through each files in a directory"""
    for dirpath, dirs, files in os.walk(folder):
        for fn in files:
            yield os.path.abspath(os.path.join(dirpath, fn))


def download_url_file(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify {filename}. Can you get to it with a browser?')
    return filename


def delete_files(dirpath: str, file_ext='*.txt'):
    exception_files = list()
    for root_dir, sub_dir, file_name in os.walk(dirpath):
        for file in fnmatch.filter(file_name, file_ext):
            try:
                print('deleted : ', os.path.join(root_dir, file))
                os.remove(os.path.join(root_dir, file))
            except Exception as err:
                print(
                    f'{err} while deleting file :', os.path.join(
                        root_dir, file)
                )
                exception_files.append(os.path.join(root_dir, file))
    if len(exception_files) != 0:
        return exception_files
    else:
        return None


class GoogleDriveDownloader:
    DRIVE_URL = 'https://docs.google.com/uc?export=download'

    def __init__(
        self,
        file_id: str = None,
        filename: str = None,
        dirpath: str = "data",
        chunk_size: int = 32_768,
    ):
        """Download files from a google drive.

        Parameters:
        ----------
        `file_id` (str, default=None):
            The id from from a shareable google driver link. An id
            should be of string length == 33.

        `destination` (str, default=None):
            The destination file on your disk.

        """
        self.file_id = file_id
        self.filename = filename
        self.dirpath = dirpath
        self.chunk_size = chunk_size
        # a description for the download progress bar.
        self.desc: str = None
        self._destination: str = None

    def _confirm_token(self, response: Any) -> Any:
        for (key, val) in response.cookies.items():
            if key.startswith('download_warning'):
                return val
        return None

    def _save_content(self, response: Any) -> IO:
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        desc = self.desc if self.desc else "Downloading... "
        self._destination = os.path.join(self.dirpath, self.filename)
        with open(self._destination, 'wb') as f:
            for chunk in tqdm(response.iter_content(self.chunk_size),
                              desc=desc):
                if chunk:
                    f.write(chunk)

    def download(self, file_id: str = None, stream: bool = True) -> IO:
        file_id = file_id if file_id else self.file_id
        session = requests.Session()
        response = session.get(
            self.DRIVE_URL, params={'id': file_id}, stream=stream)
        token = self._confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(
                self.DRIVE_URL, params=params, stream=stream)
        self._save_content(response)
        logger.info("File saved in path, %s", self._destination)


class File:
    """File writer and reader for text and jsonl files.

    Parameters:
    -----------
    `path` (str, default="data"):
        The default path directory to use in a session.

    Usage:
        >>> file = File("data")
        >>> file.write_jsonl(Iterable, "file_name.jsonl")
        # this also works:
        >>> File.write_jsonl(Iterable, "file_name.jsonl", path='downloads')

    """
    _path = None

    def __init__(self, path: str = "data") -> IO:
        self.path = path
        if self.path:
            File._path = self.path

    @classmethod
    def _make_dirs(cls, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    @classmethod
    def write_text(cls, doc: List[str], name: str, path: str = None) -> TextIO:
        path = path if path else File._path
        cls._make_dirs(path)
        with open(os.path.join(path, name), "w", encoding="utf-8") as f:
            for line in doc:
                if len(line.strip()) > 0:
                    f.write("%s\n" % line)

    @classmethod
    def read_text(cls, path: str, lines: bool = True) -> Generator:
        """Reads TEXT files (recommend leaving lines as True for large files).

        Parameters:
        -----------
        `lines` (bool, default=True):
            If False, all data is read in at once; otherwise, data is read in
            one line at a time.
        """
        with open(path, "r", encoding="utf-8") as f:
            if lines is False:
                yield f.read()
            else:
                for line in f:
                    yield line

    @classmethod
    def write_jsonl(
            cls,
            doc: List[Text],
            name: str,
            path: str = None,
            text_only: bool = True,
    ) -> IO:
        """Write to a jsonl file

        Parameters:
        ----------

        `text_only` (bool, default=True):
            If true only one dict key is present. If more than one key
            set text_only to false.

        """
        path = path if path else File._path
        cls._make_dirs(path)
        with io.open(os.path.join(path, name), "w", encoding="utf-8") as f:
            for line in doc:
                if text_only:
                    print(json.dumps(
                        {"text": line}, ensure_ascii=False), file=f)
                else:
                    print(json.dumps(line, ensure_ascii=False), file=f)

    @classmethod
    def read_jsonl(cls, path: str, lines: bool = True) -> Generator:
        """Reads JSONL files (recommend leaving lines as True for large files).

        Parameters:
        -----------
        `lines` (bool, default=True):
            If False, all data is read in at once; otherwise, data is read in
            one line at a time.
        """
        with open(path, mode="r", encoding="utf-8") as f:
            if lines is False:
                yield json.loads(f)
            else:
                for line in f:
                    yield json.loads(line)
