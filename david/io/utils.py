import fnmatch
import io
import json
import os
import urllib

import pandas as pd
import requests


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

    def __init__(self, file_id: str, file_name='temp.p',
                 dirpath='data', chunk_size=32_768):
        self.file_id = file_id
        self.file_name = file_name
        self.dirpath = dirpath
        self.chunk_size = chunk_size
        self.destination = os.path.join(self.dirpath, file_name)

    def download_df(self, stream=True, load_df=False):
        session = requests.Session()
        response = session.get(
            self.DRIVE_URL,
            params={'id': self.file_id},
            stream=stream)
        token = self._confirm_token(response)
        if token:
            params = {'id': self.file_id, 'confirm': token}
            response = session.get(
                self.DRIVE_URL,
                params=params,
                stream=stream)
        self._save_content(response)
        if load_df:
            return self._load_df()

    def _confirm_token(self, response):
        for (key, val) in response.cookies.items():
            if key.startswith('download_warning'):
                return val
        return None

    def _save_content(self, response):
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        with open(self.destination, 'wb') as f:
            for chunk in response.iter_content(self.chunk_size):
                if chunk:
                    f.write(chunk)

    def _load_df(self):
        df = pd.read_pickle(self.destination)
        return df


class File:
    """File reader and writer for TEXT and JSON files."""

    def __init__(self, path="data"):
        self.path = path

    def _make_dirs(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def write_text(self, doc: list, name, path=None):
        path = path if path else self.path
        self._make_dirs(path)
        with open(os.path.join(path, name), "w", encoding="utf-8") as f:
            for line in doc:
                if len(line.strip()) > 0:
                    f.write("%s\n" % line)

    def read_text(self, path, lines=True):
        """Reads TEXT files (recommend leaving lines as True for large files).

        Args:
            lines (bool, default=True): If False, all data is read
                in at once; otherwise, data is read in one line at a time.
        """
        with open(path, "r", encoding="utf-8") as f:
            if lines is False:
                yield f.read()
            else:
                for line in f:
                    yield line

    def write_jsonl(self, doc, name, path=None, text_only=True):
        path = path if path else self.path
        self._make_dirs(path)
        with io.open(os.path.join(path, name), "w", encoding="utf-8") as f:
            for line in doc:
                if text_only:
                    print(json.dumps(
                        {"text": line}, ensure_ascii=False), file=f)
                else:
                    print(json.dumps(line, ensure_ascii=False), file=f)

    def read_jsonl(self, path, lines=True):
        """Reads JSONL files (recommend leaving lines as True for large files).

        Args:
            lines (bool, default=True): If False, all data is read
                in at once; otherwise, data is read in one line at a time.
        """
        with open(path, mode="r", encoding="utf-8") as f:
            if lines is False:
                yield json.loads(f)
            else:
                for line in f:
                    yield json.loads(line)
