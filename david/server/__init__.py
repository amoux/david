import os

import records

COUNT_QUERIES = {
    'videos': 'SELECT DISTINCT video_id FROM comments;',
    'comments': 'SELECT video_id, COUNT(*) c FROM comments GROUP BY video_id;'}


class CommentsDB(records.Database):

    BASE_SQL_URL = os.environ.get('DAVID_COMMENTS_DB')
    BASE_DB_NAME = 'comments_v2.db'

    def __init__(self, sql_url=None, table_name=None):
        self.sql_url = sql_url
        self.table_name = table_name

        if not self.sql_url:
            self.sql_url = 'sqlite:///{}'.format(
                os.path.join(self.BASE_SQL_URL, self.BASE_DB_NAME))

        super().__init__(self.sql_url)
        if not self.table_name:
            self.table_name = self.get_table_names()[0]

    def search_comments(self, text_pattern: str):
        """Query comments based on word patterns.

        Example: text_pattern='%make a video%'
        """
        return self.query("select text from {} where text like '{}'".format(
            self.table_name, text_pattern))

    def get_all_comments(self):
        return self.query(f'select text from {self.table_name}')

    def query_comments_by_id(self, id_or_ids, id_table='id'):
        get = {'row': 'is {}', 'rows': 'between {} and {}'}
        condition = ''
        if isinstance(id_or_ids, list) and len(id_or_ids) == 2:
            condition = get['rows'].format(id_or_ids[0], id_or_ids[1])

        elif isinstance(id_or_ids, int) or len([id_or_ids]) == 1:
            condition = get['row'].format(
                id_or_ids if isinstance(id_or_ids, int) else id_or_ids[0])

        return self.query('select text from {} where {} {}'.format(
            self.table_name, id_table, condition))
