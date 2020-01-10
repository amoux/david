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


class File(object):
    basepath = "data"
    
    @staticmethod
    def directory(basepath: str = "data"):
        File.basepath = basepath

    @classmethod
    def _loadpath(cls, file_name: str):
        if not os.path.isdir(cls.basepath):
            os.makedirs(cls.basepath, exist_ok=True)
        return os.path.join(cls.basepath, file_name)

    @classmethod
    def write_text(cls, doc: List[str], name: str) -> TextIO:
        file_path = cls._loadpath(name)
        with open(file_path, "w", encoding="utf-8") as file:
            for line in doc:
                if len(line.strip()) > 0:
                    file.write("%s\n" % line)

    @classmethod
    def read_text(cls, name: str, lines=True) -> Generator:
        file_path = cls._loadpath(name)
        with open(file_path, "r", encoding="utf-8") as file:
            if lines is False:
                yield file.read()
            else:
                for line in file:
                    yield line

    @classmethod
    def write_jsonl(cls, doc: List[Text], name: str, text_only=True) -> IO:
        file_path = cls._loadpath(name)
        with io.open(file_path, "w", encoding="utf-8") as file:
            for line in doc:
                if text_only:
                    print(json.dumps({"text": line}, ensure_ascii=False), file=file)
                else:
                    print(json.dumps(line, ensure_ascii=False), file=file)

    @classmethod
    def read_jsonl(cls, name: str, lines: bool = True) -> Generator:
        file_path = cls._loadpath(name)
        with open(file_path, mode="r", encoding="utf-8") as file:
            if lines is False:
                yield json.loads(file)
            else:
                for line in file:
                    yield json.loads(line)
                   
    def __repr__(cls):
        return f"File< {os.listdir(File.basepath)} >"
    
    __call__ = directory
