from functools import namedtuple
from os import environ
from os.path import join
from typing import AnyStr, Collection, Iterable, List, Mapping, Sequence

from records import Database

Uniques = {
    'videos': 'SELECT DISTINCT video_id FROM comments;',
    'comments': 'SELECT video_id, COUNT(*) c FROM comments GROUP BY video_id;',
}

SimilarTextsInColumn = ("SELECT text FROM {table} "
                        "WHERE {id_col} {condition} "
                        "AND text LIKE '{text}';")
condition = {
    'rm_condition': ' {condition}',
    'is_row': ' is {row}',
    'between_ab': ' between {row_a} and {row_b}',
}


def pointer(name: str, params: dict):
    name = namedtuple(name, params.keys())
    return name(*params.values())


def prep_search(text: str, joinby=' ', outer='%'):
    '''Cleans a given sequence and returns
    a formatted pattern.
    '''
    text = ' '.join(text.split()).replace(' ', joinby)
    return (f'{outer}{text}{outer}')


def format_condition(template: str, doc_size: list = None, condition=condition):
    '''Sql statement template formatter.

    Parameters:
    ----------

    `template`: (type=str)
        A sql statement-like pattern to for a clause
        condition to use.

    `doc_size` : (types=List[str|int] or [NoneType])
        A clause can be a sigle or pair-combinations
        or None. For example, passing the doc_size=[1,10]
        returns a formatted str: 'BETWEEN 1 AND 100'.
    '''
    Condition = pointer('ConditionClause', condition)
    if not doc_size:
        return template.replace(Condition.rm_condition, '')
    else:
        if len(doc_size) == 1:
            con = Condition.is_row.format(row=doc_size[0])
        if len(doc_size) == 2:
            con = Condition.between_ab.format(row_a=doc_size[0],
                                              row_b=doc_size[1])
        return template.replace(Condition.rm_condition, con)


def sequence_mapping(table: str,
                     id_col: str,
                     text: Sequence[str],
                     joinby: str = '_',
                     outer: str = '%',
                     doc_size: List[str] = None,
                     ) -> Sequence[Mapping]:
    '''Returns a searchable sequence mapping.
    '''
    if doc_size and not isinstance(doc_size, list):
        doc_size = [doc_size]
    similar = format_condition(SimilarTextsInColumn, doc_size=doc_size)
    return similar.format(table=table, id_col=id_col,
                          text=prep_search(text, joinby, outer))


def get_similartexts(words: Sequence[str],
                     table: str = 'comments',
                     id_col: AnyStr = 'id',
                     joinby: str = '%_%',
                     outer: str = '%',
                     doc_size: List[int] = None,
                     as_list: bool = False,
                     db_name: str = 'comments.db'
                     ) -> Iterable[Collection]:
    '''
    Get a document containing texts matching
    a given sequence.

    Parameters:
    ----------

    `text`: (type=str)
        A sring sequence or key-words to use for extracting
        similar sequences from any maching row.

    `doc_size` : (types=List[str|int] or [NoneType])
        A clause can be a sigle or pair-combinations
        or None. For example, passing the doc_size=[1,10]
        returns a formatted str: 'BETWEEN 1 AND 100'.

    `db_name`: (str)
        The name of the dataset to load from DAVID_COMMENTS_DB.
        Available dataset names: `comments.db,  comments_v1.db`

    Usage:
    -----

        >>> max_rows = 252_848 # this is an optional limiter.
        >>> text = 'make a video'
        >>> docs = get_similartexts(text, doc_size=[1, max_rows])
        >>> for doc in docs: print(doc.text)
        ...
        'thanks for a great video of introduction to Tensorflow 👍.'
        'Hello TechLead, Can you make a video about Scala in software engineer?'

    '''
    sqlpath = 'sqlite:///{}'.format(
        join(environ.get('DAVID_COMMENTS_DB'), db_name)
    )
    records = Database(sqlpath)
    similar_docs = records.query(
        sequence_mapping(
            table=table,
            id_col=id_col,
            text=words,
            doc_size=doc_size,
            joinby=joinby,
            outer=outer
        )
    )
    if as_list:
        return [' '.join(doc.text.split()) for doc in similar_docs]
    else:
        return similar_docs
