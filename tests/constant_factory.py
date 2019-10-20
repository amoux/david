from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple, Type


def constant_factory_str(value):
    """
    >>> d = defaultdict(constant_factory('<missing>'))
    >>> d.update(name='John', action='ran')
    >>> '%(name)s %(action)s to %(object)s' % d
    [Out] 'John ran to <missing>'
    """
    return lambda: value


def _lists(collection):
    # lists grouping factory.
    d = defaultdict(list)
    for k, v in collection:
        d[k].append(v)
    return d


def _sets(collection):
    # sets builder factory.
    d = defaultdict(set)
    for k, v in collection:
        d[k].add(v)
    return d


def _ints(collection):
    # ints counting factory.
    d = defaultdict(int)
    for k in collection:
        d[k] += 1
    return d


def constant_factory(collection: List[Tuple],
                     func: Type[Set[List[int]]],
                     sort_items=False) -> DefaultDict[Tuple]:
    """Dictionary Factory Builder.

    Parameters:
    ----------
    collection : object -> list[tuples]
    by_func : type -> [set|list|int]
    sort_items : (bool)


    func : object[Types]:
    --------------------

        * set  -  Building a dictionary of sets.
        * int  -  Counting a frequecy, e.g. a count freq of sequences.
        * list -  Grouping a sequence of key-value pairs into a dict-of-lists

        >>> constant_factory([('blue', 4), ('blue', 2)], func=list)
        ... [('blue', [2, 4])

        >>> S = ['python is A', 'python is A', 'java is D', 'java is C']
        >>> constant_factory(S, int, sort_items=True)
        ... [('java is C', 1), ('java is D', 1), ('python is A', 2)]
    """
    if func is set:
        collection = _sets(collection)
    elif func is int:
        collection = _ints(collection)
    elif func is list:
        collection = _lists(collection)
    if sort_items:
        return sorted(collection.items())
    else:
        return collection
