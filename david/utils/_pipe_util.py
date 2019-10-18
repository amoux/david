import time
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
