import json
import os
import sqlite3

import records

COUNT_QUERIES = {
    'videos': 'SELECT DISTINCT video_id FROM comments;',
    'comments': 'SELECT video_id, COUNT(*) c FROM comments GROUP BY video_id;'}


class CommentsDB(records.Database):
    """Comments Database Data Loader Component."""
    DAVID_HOME_SQLITE = os.environ.get('DAVID_COMMENTS_DB')
    DEFAULT_DB_FILE = 'comments_v2.db'

    def __init__(self, sql_url=None, db_name=None, table_name=None):
        self.sql_url = sql_url
        self.db_name = db_name
        if not self.db_name:
            self.db_name = self.DEFAULT_DB_FILE
        if not self.sql_url:
            sql_url = os.path.join(self.DAVID_HOME_SQLITE, self.db_name)
            self.sql_url = 'sqlite:///{}'.format(sql_url)
        super().__init__(self.sql_url)
        self.table_name = table_name
        if not self.table_name:
            self.table_name = self.get_table_names()[0]

    def search_comments(self, text_pattern: str):
        """Query comments based on word patterns e.g., '%make a video%'."""
        return self.query("select text from {} where text like '{}'".format(
            self.table_name, text_pattern))

    def get_all_comments(self):
        return self.query(f'select text from {self.table_name}')


class CommentsSQL(object):
    """Comments database Connector."""
    DAVID_HOME_SQLITE = os.environ.get('DAVID_COMMENTS_DB')
    LATEST_COMMENTS_DB = 'comments_v2.db'

    def __init__(self, sql_path=None, db_file_name=None, table_name=None):
        self.sql_path = sql_path
        self.db_file_name = db_file_name
        self.table_name = table_name
        if not self.db_file_name:
            self.db_file_name = self.LATEST_COMMENTS_DB
        if not self.sql_path:
            self.sql_path = os.path.join(
                self.DAVID_HOME_SQLITE, self.db_file_name)
        self.conn = sqlite3.connect(self.sql_path)
        if not self.table_name:
            cursor = self.conn.execute(
                "select name from sqlite_master where type='table' or type='view'")
            self.table_name = tuple(t[0] for t in cursor.fetchall())[0]

    @property
    def column_names(self):
        c = self.conn.execute('select * from %s limit 1' % self.table_name)
        return [col_name[0] for col_name in c.description]

    @property
    def unique_videoids(self):
        c = self.conn.execute(
            "select distinct video_id from %s limit 1" % self.table_name)
        return [vid_id[0] for vid_id in c.fetchall()]

    @property
    def num_rows(self):
        c = self.conn.execute(
            "select count(*) from %s limit 1" % self.table_name)
        return c.fetchone()[0]

    def search_comments(self, pattern):
        c = self.conn.execute(
            f"select text from {self.table_name} where text like '{pattern}';")
        return [text[0] for text in c.fetchall()]
