import json
import os

import records
from david.utils.io import as_jsonl_file as _as_jsonl_file

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

    def as_jsonl_file(self, texts, fname, output_dir='.'):
        """Write an Iterable of sequences (texts) as a JSONL file."""
        _as_jsonl_file(texts, fname, output_dir)

    def search_comments(self, text_pattern: str):
        """Query comments based on word patterns e.g., '%make a video%'"""
        return self.query("select text from {} where text like '{}'".format(
            self.table_name, text_pattern))

    def get_all_comments(self):
        return self.query(f'select text from {self.table_name}')
