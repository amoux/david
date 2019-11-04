import json
import os

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

    def as_jsonl(self, doc_obj, fname, output_dir='data'):
        """Write a search query iterable to file as JSONL format.

        Usage:
        -----
            >>> db = CommentsDB()
            >>> comments = db.search_comments("%make a video%")
            >>> db.as_jsonl(comments, 'comments.jsonl')
        """
        if isinstance(doc_obj, CommentsDB.__class__.__base__):
            is_valid_obj = doc_obj.as_dict()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, fname)
        with open(file_path, 'w', encoding='utf-8') as jsonl_file:
            for line in is_valid_obj:
                json.dump(line, jsonl_file)
                jsonl_file.write('\n')

    def search_comments(self, text_pattern: str):
        """Query comments based on word patterns e.g., '%make a video%'"""
        return self.query("select text from {} where text like '{}'".format(
            self.table_name, text_pattern))

    def get_all_comments(self):
        return self.query(f'select text from {self.table_name}')
