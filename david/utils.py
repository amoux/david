import os
import shutil
import time

import torch


def current_path(dirname: str):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), dirname))


def get_data_home(data_home: str = None):
    """Return the path of the david_data directory.

    Args:
        data_home (str, default=None): The path to david_data data dir.
    """
    if not data_home:
        data_home = os.environ.get(
            'DAVID_DATA', os.path.join('~', 'david_data'))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def clear_data_home(data_home: str = None):
    """Delete all the content of the data home cache.

    Args:
        data_home (str, default=None): The path to david_data data dir.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)


def is_cuda_enabled(torch=torch, emptycache=False):
    """Prints GPU device and memory usage information if Cuda is enabled."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(device))
        print('\nmemory usage:')
        print(f"allocated: {torch.cuda.memory_allocated(device)} GB")
        print(f"cached: {torch.cuda.memory_cached(device)} GB")
    if emptycache:
        torch.cuda.empty_cache()


def timeit(method):
    """Timer Decorator Utility."""
    def timed(*args, **kw):
        t1 = time.time()
        result = method(*args, **kw)
        t2 = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((t2 - t1) * 1000)
        else:
            print('( {!r} ) took: {:2.2f} ms'.format(
                method.__name__, (t2 - t1) * 1000))
        return result
    return timed
