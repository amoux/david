import json
import os
import sqlite3
from collections import namedtuple
from typing import (KT, Any, AnyStr, Generator, Iterable, List, Mapping,
                    Optional, Tuple)

from ..datasets import GDRIVE_SQLITE_DATABASES, download_sqlite_database

COUNT_QUERIES = {
    'videos': "select distinct video_id from comments;",
    'comments': "select video_id, count(*) c from {} group by video_id;"
}


def check_file_exists(file_name: str, file_path: str,
                      file_ext: str = ".db") -> Any:
    """Validates if the file exists in the directory.
    Returns the file name if exists. Otherwise it returns None.
    Usage:
        >>> check_file_exists('unbox', david_home_path)
        'lew_comments_unbox.db'
    """
    file_found = None
    for file in os.listdir(file_path):
        if file.endswith(file_ext) \
                and file.replace(file_ext, "").endswith(file_name):
            file_found = file  # get the first occurrence for now:
    return file_found


def db_file_exist(file: str, path: str) -> Any:
    file_path = os.path.join(path, check_file_exists(file, path))
    if not os.path.isfile(file_path):
        return None
    return file_path


class CommentsSql(object):
    DAVID_SQLITE = os.path.join(os.environ.get('DAVID_DATA'), "sqlite")

    def __init__(self,
                 sql_file: Optional[str] = None,
                 sql_path: Optional[str] = None,
                 table_name: Optional[str] = None):
        """Database connector for collections of YouTube comments.

        Parameters:
        ----------

        `sql_file` (str, default=None):
            Choose a database name to load, currently available: ('v1', 'v2',
            'unbox'). If the file does not exist it will download the file
            from the server automatically.

        `sql_path` (str, default=None):
            Pass the full path where the database file is located.

        """
        self.sql_file = sql_file
        self.sql_path = sql_path
        self.table_name = table_name

        if self.sql_file and sql_path is None:
            if not os.path.isdir(self.DAVID_SQLITE):
                os.makedirs(self.DAVID_SQLITE, exist_ok=True)
            temp_sqlfile, sqlite_home = (self.sql_file, self.DAVID_SQLITE)
            if temp_sqlfile in GDRIVE_SQLITE_DATABASES.keys():
                try:
                    sql_filepath = db_file_exist(temp_sqlfile, sqlite_home)
                except TypeError:
                    sql_filepath = download_sqlite_database(
                        temp_sqlfile, sqlite_home, return_destination=True)
                # update the evaluated path to the db file and connect.
                self.conn = sqlite3.connect(sql_filepath)
                self.sql_path = sql_filepath
                self.sql_file = os.path.basename(sql_filepath)
                del sql_filepath
                del temp_sqlfile

            else:
                raise ValueError("{}, Error is not a valid file to load \
                select one from: {}.".format(self.sql_file,
                                             GDRIVE_SQLITE_DATABASES.keys()))

        if self.sql_path and self.sql_file is None:
            self.conn, self.sql_file = sqlite3.connect(
                self.sql_path), os.path.basename(self.sql_path)

        if self.table_name is None:
            cursor = self.conn.execute(
                "select name from sqlite_master \
                    where type='table' or type='view'")
            self.table_name = tuple(t[0] for t in cursor.fetchall())[0]

    @property
    def column_names(self) -> Iterable[List[str]]:
        c = self.conn.execute('select * from %s limit 1' % self.table_name)
        return [col_name[0] for col_name in c.description]

    @property
    def unique_videoids(self) -> Iterable[List[Tuple[str, ...]]]:
        c = self.conn.execute(
            "select video_id, count(*) c from {} group by video_id;".format(
                self.table_name))
        return [video_id_counts for video_id_counts in c.fetchall()]

    @property
    def num_rows(self) -> int:
        c = self.conn.execute(
            "select count(*) from %s limit 1" % self.table_name)
        return c.fetchone()[0]

    def fetch_comments(
            self, pattern: str,
            columns: str = "text",
            sort_col: Optional[str] = None,
            reverse: bool = True,
    ) -> Iterable[List[Mapping[Tuple[KT, ...], Generator]]]:
        """Get a batch of of comments based on the key word patters:

        Parameters:
        -----------

        `pattern` (str):
            A pattern in the form %key key% or %<K>% %<K>%.

        `column_names` (str, default="text"):
            Pass the name of the columns to you want to load separated by
            `,` punctuation. For example,  columns="id, video_id, text".

        `sort_column` (str, default=Optional[str]):
            Pass the name of the column to sort the returning Iterable.
            It must be a single column name of string type.

        Usage:
            >>> pattern = "%make a new video about%"
            >>> doc = fetch_comments(pattern, "id, text", sort_by="id")
            >>> print(doc[0].id, doc[0].text)
            '(792286, 'Can you make a video about pixel 1 2!')'

        """
        c = self.conn.execute(
            "select {} from {} where text like '{}';".format(
                columns, self.table_name, pattern))
        # assigns the column names selected as attributes (mapping).
        mapped_attrs = namedtuple(self.table_name.capitalize(), columns)
        mapped_batch = list(map(mapped_attrs._make, c.fetchall()))
        if sort_col and isinstance(sort_col, str):
            idx = columns.replace(" ", "").split(",").index(sort_col)
            return sorted(mapped_batch, key=lambda k: k[idx], reverse=reverse)
        return mapped_batch
