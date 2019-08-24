import time
from collections import namedtuple


REGX_POINTERS = {
    'match_titles': "(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'match_quotes': '"(?:\\.|(\\")|[^""\n])*"',
    'match_times': '([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'match_youtubeurl': '(?:https?:\\/\\/)?(?:(?:(?:www\\.?)?youtube\\.com(?:\\/(?:(?:watch\\?.*?(v=[^&\\s]+).*)|(?:v(\\/.*))|(channel\\/.+)|(?:user\\/(.+))|(?:results\\?(search_query=.+))))?)|(?:youtu\\.be(\\/.*)?))',
    'trim_whitespaces': '(?:\\s)\\s")'
}

DATASET_PATHS = {
    'creator_videoid_info': "david/datasets/ycd_csv/ycc_creators_video_info.csv",
    'web_tensorboard': "david/datasets/ycd_csv/ycd_web_tensorboard.csv",
    'web_md': "david/datasets/ycd_csv/ycc_web_md.csv",
    'web_lg': "david/datasets/ycd_csv/ycc_web_lg.csv"
}


def pointer(getfrom: str):
    '''Pointer Collection Utils.
    `getfrom` : Choose one of them = ('datasets' or 'regex')
    '''

    if ('datasets') in getfrom.lower():
        named = namedtuple('YCD', DATASET_PATHS.keys())
        return named(*DATASET_PATHS.values())

    elif ('regex') in getfrom.lower():
        named = namedtuple('Regex', REGX_POINTERS.keys())
        return named(*REGX_POINTERS.values())


def timeit(method):
    '''@timeit Decorator.
    Which allows you to measure the execution time of
    the method/function by just adding the `@timeit`
    decorator on the method.

    * For more utily decorators install:
    https://pypi.org/project/profilehooks/
    >>> pip install profilehooks
    '''
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
