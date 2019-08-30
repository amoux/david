import os
import re
from pprint import pprint

import pandas as pd
from tqdm import tqdm

from david.youtube.scraper import download
from david.youtube.search import yt_channel
from david.youtube.utils import CHANNEL_STATS_API

COMMENTS = CHANNEL_STATS_API['comments']
VIDEO_ID = 'vidId'


def make_savepath(query: str, save_path: str):
    '''Creates and names the directorier per query.
    '''
    dir_name = re.sub(' ', '_', query)
    dir_path = os.path.join(save_path, dir_name)
    return dir_path


def save_corpus(df: object, query: str,
                save_path: str, join_by='_', ftype='.csv'):
    '''
    Saves the CSV file to its parent directory.
    The file is named in relation to the a search query.
    '''
    fn = re.sub(' ', join_by, query)
    fn = fn + ftype
    fp = make_savepath(query, save_path)
    fp = os.path.join(fp, fn)
    df.to_csv(fp)


def get_comments(videoids: list, fp: str, limit: int):
    '''Download youtube comments.
    Scrapes comments for each video id in the list.
    '''
    for vid in tqdm(videoids):
        download(video_id=vid, dirpath=fp, limit=limit)


def build_corpus(query,
                 max_results=10,
                 min_comments=None,
                 limit=None,
                 download=False,
                 save_path='downloads',
                 to_csv=False,
                 join_by='_'):
    '''
    Youtube Comments Courpus Builder. Downloads comments from
    a youtube video id and saves the results to a csv file.

    Parameters:
    ----------

    `query` : (str)
        A string used to find video ids by
        matching query keywords.

    `max_results` : (int)
        Number of maximum results (matching videos)
        to get for a given query.

    `min_comments` : (int)
        Min comments a video id requires, to be
        worthy of downloading.

    `download` : (bool)
        If set to False, no comments will be downloaded.

    `limit` : (int)
        Limits how many comments should be downloaded per
        video id.

    `to_csv` :
        Save search results from the Youtube Data API to
        a csv file.

    '''
    res = yt_channel(query, max_results)
    df = pd.DataFrame.from_dict(res, orient="columns")
    df[COMMENTS] = df[COMMENTS].astype(int)
    df = df[df[COMMENTS] > min_comments]
    videoid = df[VIDEO_ID].values.tolist()

    if download:
        get_comments(videoid, save_path, limit)

    if to_csv:
        save_corpus(df, query, save_path, join_by)

    elif not (download and to_csv):
        # prints the response if both False.
        pprint(res)


if __name__ == '__main__':

    QSEARCH = 'python is the future'
    build_corpus(QSEARCH, max_results=10,
                 min_comments=2000, download=True,
                 limit=None, to_csv=True)
