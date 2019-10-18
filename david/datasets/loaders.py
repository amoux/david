import os
from os import environ, makedirs
from os.path import dirpath, exists, expanduser, join

import pandas as pd
import requests
from sklearn.datasets import load_files as _load_files
from sklearn.datasets import _clear_data_home as _clear_data_home
from sklearn.utils import Bunch as _Bunch


def load_comments_lg():
    module_path = dirpath(__file__)
    filename = join(module_path, 'ycd_csv', 'ycc_web_lg.csv')
    with open(join(module_path, 'description', 'ytc_lg.rst')) as rst_file:
        file_descr = rst_file.read()
    return Bunch(DESCR=file_descr, filename=filename)


def get_data_home(data_home: str = None):
    """Return the path of the david_data directory.

    This folder is used by some large dataset loaders to avoid downloading
    the data several times. By default the data dir is set to a folder named
    'david_data' in the user home folder. Alternatively, it can be set by
    the 'DAVID_DATA' environment variable or programmatically by giving an
    explicit folder path. The '~' symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.

    Parameters:
    ----------

    `data_home` : (str, default=None)
        The path to david_data data dir.

    """
    if data_home is None:
        data_home = environ.get('DAVID_DATA', join('~', 'david_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def remove_data_home(data_home=None):
    """Calling this method deletes the root directory `david_data`
    including all the datasets.
    """
    _clear_data_home(data_home=data_home)


def load_datasets(container_path, description=None, categories=None,
                  load_content=True, shuffle=True, encoding=None,
                  decode_error='strict', random_state=0):
    """Load dataset files with categories as subfolder names.

    - Return all categories and its files:

        >>> ycd_json = join(get_data_home(), 'ycd_json')
        >>> bunch = load_datasets(ycd_json, load_content=False)
        >>> bunch.filenames # list of all filenames
        >>> bunch.target_names # list of all categories.
        >>> bunch.target # index array

    - Return all files within a category. Pass the name of the
    category e.g. 'worst_car_trends' where the name of the category
    is the directory name:

        >>> ycd_json = join(get_data_home(), 'ycd_json')
        >>> file_names = load_datasets(
                                ycd_json,
                                categories='worst_car_trends',
                                load_content=False)
    Parameters:
    ----------

    `container_path` : string or unicode
        Path to the main folder holding one subfolder per category.

    `description` : string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    `categories` : A collection of strings or None, optional (default=None)
        If None (default), load all the categories. If not None, list of
        category names to load (other categories ignored).

    `load_content` : boolean, optional (default=True)
        Whether to load or not the content of the different files. If true a
        'data' attribute containing the text information is present in the data
        structure returned. If not, a filenames attribute gives the path to the
        files.

    Returns
    -------

    data : Bunch Dictionary-like object.
    """
    return _load_file(container_path, description, categories,
                      load_content, shuffle, encoding,
                      decode_error, random_state)


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
