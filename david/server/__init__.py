import json
import os
import sqlite3
from typing import List, Any, Optional
import records

from ..datasets import GDRIVE_SQLITE_DATABASES, download_sqlite_database

COUNT_QUERIES = {
    'videos': 'SELECT DISTINCT video_id FROM comments;',
    'comments': 'SELECT video_id, COUNT(*) c FROM comments GROUP BY video_id;'}


def check_file_exists(
        file_name: str,
        file_path: str,
        file_ext: str = ".db") -> Any:
    """Validates if the file exists in the directory.

    Usage:
        >>> check_file_exists('unbox', david_home_path)
        'lew_comments_unbox.db'

    Returns the file name if exists. Otherwise it returns None.
    """
    file_found = None
    for file in os.listdir(file_path):
        if file.endswith(file_ext) and file.replace(
                file_ext, "").endswith(file_name):
            file_found = file  # get the first occurrence for now:
    return file_found


def is_file_in_dirpath(file: str, path: str) -> Any:
    file_path = os.path.join(path, check_file_exists(file, path))
    if not os.path.isfile(file_path):
        return None
    return file_path


class CommentsSQL(object):

    DAVID_HOME_SQLITE = os.environ.get('DAVID_COMMENTS_DB')
    # create one/both directories if missing: david_home/sqlite.
    if not os.path.exists(DAVID_HOME_SQLITE):
        os.makedirs(DAVID_HOME_SQLITE)
    AVAILABLE_DATABASES = GDRIVE_SQLITE_DATABASES

    def __init__(
            self,
            sql_file: Optional[str] = None,
            sql_path: Optional[str] = None,
            table_name: Optional[str] = None,
    ):
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

        # Downloads the database file if it doesn't exists.
        if self.sql_file and sql_path is None:
            temp_sqlfile, sqlite_home = (
                self.sql_file, self.DAVID_HOME_SQLITE)
            if temp_sqlfile in self.AVAILABLE_DATABASES.keys():
                try:
                    # If the file doesn't exists download the file:
                    sql_file_path_exist = is_file_in_dirpath(
                        temp_sqlfile, sqlite_home)
                except TypeError:
                    # sql file does not exist, download the database.
                    sql_file_path_exist = download_sqlite_database(
                        temp_sqlfile, sqlite_home, return_destination=True)
                # sql file path loaded, connect to the database.
                self.conn = sqlite3.connect(sql_file_path_exist)
                self.sql_file, self.sql_path = os.path.basename(
                    sql_file_path_exist), sql_file_path_exist
            else:
                raise ValueError("{}, Error is not a valid file to load \
                select one from: {}.".format(
                    self.sql_file, self.AVAILABLE_DATABASES.keys()))

        if self.sql_path and self.sql_file is None:
            self.conn, self.sql_file = sqlite3.connect(
                self.sql_path), os.path.basename(self.sql_path)

        if self.table_name is None:
            cursor = self.conn.execute("select name from sqlite_master \
                where type='table' or type='view'")
            self.table_name = tuple(t[0] for t in cursor.fetchall())[0]

    @property
    def column_names(self) -> List[str]:
        c = self.conn.execute('select * from %s limit 1' % self.table_name)
        return [col_name[0] for col_name in c.description]

    @property
    def unique_videoids(self) -> List[str]:
        c = self.conn.execute(
            "select distinct video_id from %s limit 1" % self.table_name)
        return [vid_id[0] for vid_id in c.fetchall()]

    @property
    def num_rows(self) -> int:
        c = self.conn.execute(
            "select count(*) from %s limit 1" % self.table_name)
        return c.fetchone()[0]

    def search_comments(self, pattern: str) -> List[str]:
        c = self.conn.execute(
            f"select text from {self.table_name} where text like '{pattern}';")
        return [text[0] for text in c.fetchall()]
