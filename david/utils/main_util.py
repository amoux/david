from functools import namedtuple

import torch

REGEX_DICT = {
    'match_titles': "(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'match_quotes': '"(?:\\.|(\\")|[^""\n])*"',
    'match_times': '([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'match_youtubeurl': '(?:https?:\\/\\/)?(?:(?:(?:www\\.?)?youtube\\.com(?:\\/(?:(?:watch\\?.*?(v=[^&\\s]+).*)|(?:v(\\/.*))|(channel\\/.+)|(?:user\\/(.+))|(?:results\\?(search_query=.+))))?)|(?:youtu\\.be(\\/.*)?))',
    'trim_whitespaces': '(?:\\s)\\s")'
}


def pointer(n: str, params: dict):
    """Returns a dictionary-like object from its key, value pairs."""
    p = namedtuple(n, params.keys())
    return p(*params.values())


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
