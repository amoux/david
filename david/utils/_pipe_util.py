import time
from collections import namedtuple
from os import makedirs
from os.path import exists, join


def timeit(method):
    """Timer Decorator Utility.

    Which allows you to measure the execution time of
    the method/function by just adding the `@timeit`
    decorator on the method.

    * For more utily decorators install:
    https://pypi.org/project/profilehooks/
    >>> pip install profilehooks
    """
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


def text2file(fn: str, docs: list, dirpath='output') -> None:
    if not exists(dirpath):
        makedirs(dirpath)
    with open(join(dirpath, fn), 'w', encoding='utf-8') as f:
        for doc in docs:
            if len(doc) > 1:
                f.write('%s\n' % doc)
        f.close()


_YOUTUBE_TAGS_REGEX_MATCHER = {
    'titles': r"(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'quotes': r'"(?:\\.|(\\")|[^""\n])*"',
    'times': r'([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'username_v1': r'\B(\@[a-zA-Z_0-9]+\b)(?!;)',
    'username_v2': r'(\@[a-zA-Z0-9_%]*)'}

_YOUTUBE_URLS_REGEX_MATCHER = {
    'videoId': r'v=([a-zA-Z0-9\_\-]+)&?',
    'vid_url1': r'youtube.[a-z]+/[a-z\?\&]*v[/|=](\w+)',
    'any_vidUrl': r'(?:https?:\/\/)?(?:www\.)?youtu(.be\/|be\.com\/watch\?v=)(.{8,})',
    'vid_url2': r'(((\?v=)|(\/embed\/)|(youtu.be\/)|(\/v\/)|(\/a\/u\/1\/))(.+?){11})'}


RegexMatchUrls = pointer('RegexMatchUrls', _YOUTUBE_URLS_REGEX_MATCHER)
RegexMatchTags = pointer('RegexMatchTags', _YOUTUBE_TAGS_REGEX_MATCHER)
