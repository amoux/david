import os
import re

import pandas as pd
from tqdm import tqdm
from pprint import pprint

from david.youtube.search import get_video_content
from david.youtube.scraper import download_comments


def mkdir_fromquery(query):
    """Names and creates a unique directory per search query
    """
    dirname = re.sub(' ', '_', query)
    dirpath = os.path.join('downloads', dirname)
    return dirpath


def save_tocsv(df, query):
    """Saves the CSV file to its parent directory.
    The file is named in relation to the a search query
    """
    fname = re.sub(' ', '_', query)
    fname = fname + '.csv'
    dirpath = mkdir_fromquery(query)
    fpath = os.path.join(dirpath, fname)
    df.to_csv(fpath)


def download_videoids(path: str, videoids: list, query: str, limit=None):
    """Downloads youtube comments from a custom WebScraping script.
    Scrapes comments from each video id in the list.
    """
    dirpath = mkdir_fromquery(query)
    for vidid in tqdm(videoids):
        download_comments(path, vidid, limit)


def from_queryresponse(
    query='',
    maxresults=10,
    min_comments=None,
    download=True,
    limit=None,
    to_csv=False
):
    """Downloads Youtube comments by Video ID from a given search query

    PARAMETERS
    ----------
    `query` : (str)
        A string used to find video ids by matching query keywords

    `maxresults` : (int)
        Number of maximum results (matching videos) to get for a given query

    `min_comments` : (int)
        Limits the videos that should be downloaded by N comments. For example,
        setting min_comments=1000, will download from videos with 1000 comments
        and greater. If None, then any video regardles of N comments for any ID

    `download` : (bool)
        If set to False, no comments will be downloaded

    `limit` : (int)
        Limits how many comments should be downloaded per video id

    `to_csv` : (bool)
        Save search results from the Youtube Data API to a csv file
    """
    response = get_video_content(query, maxresults)
    df = pd.DataFrame.from_dict(response, orient="columns")
    df['commentCount'] = df['commentCount'].astype(int)
    # get only videos with comment len > 2000
    df = df[df['commentCount'] > min_comments]
    # save results to the downloads directory
    videoids = df['vidId'].values.tolist()

    if download:
        download_videoids(videoids, query, limit=limit)
    if to_csv:
        save_tocsv(df, query)
    elif not (download and to_csv):
        pprint(response)


SEARCH_QUERY = 'company china trump world american'


def main():
    from_queryresponse(
        SEARCH_QUERY,
        maxresults=20,
        min_comments=2000,
        download=True,
        limit=None,
        to_csv=True)


if __name__ == '__main__':
    main()
