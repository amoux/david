"""Server Configuration variables."""
import os


BASE_USER_DB_PATH = 'sqlite:///{}'.format(
    os.path.join(os.path.dirname(__file__), 'vuepoint_test.db'))
BASE_COMMENTS_DB_PATH = 'sqlite:///{}'.format(
    os.path.join(os.path.dirname(__file__), 'comments.db'))
