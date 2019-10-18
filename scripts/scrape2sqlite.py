import datetime
import sys
from sys import stdout

import dataset
from david.youtube import scraper

sys.path.append('../')


FILE = 'pending_downloads.txt'
DB_SQL = 'sqlite:///yt_comments.db'
DB_TABLE = 'comments'

today = datetime.datetime.now()
today = '{:%Y-%m-%d %H:%M:%S}'.format(today)
db = dataset.connect(DB_SQL)
table = db.create_table(DB_TABLE)
table.create_column('timestamp', db.types.datetime)


def get_videos(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        videos = []
        for line in f.readlines():
            videos.append(line.strip('\n'))
        return videos


def download_comments(videoid: str, table: str, sqlite_url: str):
    count = 0
    with dataset.connect(sqlite_url) as sqlite:
        for comment in scraper._scrape_comments(videoid):
            sqlite[table].insert(dict(
                cid=comment['cid'], text=comment['text'],
                time=comment['time'], author=comment['author'],
                video_id=videoid))

            count += 1
            stdout.write('mining %d comment(s)\r' % count)
            stdout.flush()


def batch_downloader(fp, table, sqlite_url):
    video_ids = get_videos(fp)
    for video in video_ids:
        download_comments(video, table, sqlite_url)


if __name__ == '__main__':
    batch_downloader(FILE, DB_TABLE, DB_SQL)
