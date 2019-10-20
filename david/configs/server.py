"""Server Configuration variables
"""
import os

BASE_USER_DB_PATH = os.path.join(os.path.dirname(__file__), 'vuepoint_test.db')
BASE_COMMENTS_DB_PATH = os.path.join(os.path.dirname(__file__), 'comments.db')

base_user_db_url = 'sqlite:///{}'.format(BASE_USER_DB_PATH)
base_comments_db_url = 'sqlite:///{}'.format(BASE_COMMENTS_DB_PATH)
