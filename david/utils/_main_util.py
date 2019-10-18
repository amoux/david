from functools import namedtuple

import torch
from sklearn.datasets import clear_data_home as _clear_data_home

REGEX_DICT = {
    'match_titles': "(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'match_quotes': '"(?:\\.|(\\")|[^""\n])*"',
    'match_times': '([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'match_youtubeurl': '(?:https?:\\/\\/)?(?:(?:(?:www\\.?)?youtube\\.com(?:\\/(?:(?:watch\\?.*?(v=[^&\\s]+).*)|(?:v(\\/.*))|(channel\\/.+)|(?:user\\/(.+))|(?:results\\?(search_query=.+))))?)|(?:youtu\\.be(\\/.*)?))',
    'trim_whitespaces': '(?:\\s)\\s")'
}


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
    data_home : (str, default=None)
        The path to david_data data dir.
    """
    if data_home is None:
        data_home = environ.get('DAVID_DATA', join('~', 'david_data'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def remove_data_home(data_home=None):
    """Calling this method deletes the root directory david_data
    including all the datasets.
    """
    _clear_data_home(data_home=data_home)


def is_cuda_enabled(torch=torch, emptycache=False):
    """Prints GPU device and memory usage information if Cuda is enabled.

    emptycache: Releases all unoccupied cached memory currently held by
    the caching allocator so that those can be used in other GPU application
    and visible in nvidia-smi.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
        print('\nmemory usage:')
        print(f"allocated: {torch.cuda.memory_allocated(device)} GB")
        print(f"cached: {torch.cuda.memory_cached(device)} GB")
    if emptycache:
        torch.cuda.empty_cache()
