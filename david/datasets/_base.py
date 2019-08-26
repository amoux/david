from os import environ, makedirs
from os.path import exists, expanduser, join, dirname

from sklearn.utils import Bunch
from sklearn.datasets import clear_data_home, load_files


def get_data_home(data_home=None):
    '''Return the path of the david_data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'david_data' in the
    user home folder.

    Alternatively, it can be set by the 'DAVID_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to david_data data dir.
    '''
    if data_home is None:
        data_home = environ.get('DAVID_DATA', join('~', 'david_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def remove_data_home(data_home=None):
    '''NOTE: Executing this function will delete the root david_data
    directory including all datasets!

    `data_home` : str | None
        The path to david_data dir.
    '''
    clear_data_home(data_home=data_home)


def load_datasets(container_path, description=None, categories=None,
                  load_content=True, shuffle=True, encoding=None,
                  decode_error='strict', random_state=0):
    '''Load dataset files with categories as subfolder names.

    Return all categories and its files:
    ------------------------------------

    >>> ycd_json = join(get_data_home(), 'ycd_json')
    >>> bunch = load_files(ycd_json, load_content=False)
    >>> bunch.filenames # list of all filenames
    >>> bunch.target_names # list of all categories.
    >>> bunch.target # index array

    Return all files within a category.
    -----------------------------------
    * Pass the name of the category e.g. 'worst_car_trends' where the
    name of the category is the directory name:

    >>> ycd_json = join(get_data_home(), 'ycd_json')
    >>> file_names = load_files(ycd_json, categories='worst_car_trends',
    >>                            load_content=False)

    Author Credit
    -------------
    This function loads datasets by using the `load_data`
    method from the `sklearn.datasets` module.

    If you set load_content=True, you should also specify the encoding of the
    text using the 'encoding' parameter. For many modern text files, 'utf-8'
    will be the correct encoding. If you leave encoding equal to None, then the
    content will be made of bytes instead of Unicode, and you will not be able
    to use most functions in `sklearn.feature_extraction.text`.

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category

    description : string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : A collection of strings or None, optional (default=None)
        If None (default), load all the categories. If not None, list of
        category names to load (other categories ignored).

    load_content : boolean, optional (default=True)
        Whether to load or not the content of the different files. If true a
        'data' attribute containing the text information is present in the data
        structure returned. If not, a filenames attribute gives the path to the
        files.

    Returns
    -------
    data : Bunch Dictionary-like object.

    '''
    return load_files(container_path, description, categories,
                      load_content, shuffle, encoding,
                      decode_error, random_state)


def load_comments_lg():
    module_path = dirname(__file__)
    filename = join(module_path, 'ycd_csv', 'ycc_web_lg.csv')
    with open(join(module_path, 'description', 'ytc_lg.rst')) as rst_file:
        file_descr = rst_file.read()
    return Bunch(DESCR=file_descr, filename=filename)
