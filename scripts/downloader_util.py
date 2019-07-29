import os
import re

import pandas as pd
from tqdm import tqdm
from pprint import pprint

from david.youtube.scraper import download_comments
from david.youtube.search import get_video_content


def mkdir_fromsearch(query):
    """Creates and names the directorier per query.
    """
    dirname = re.sub(' ', '_', query)
    dirpath = os.path.join('downloads', dirname)
    return dirpath


def save_tocsv(df, query):
    """Saves the CSV file to its parent directory.
    The file is named in relation to the a search query.
    """
    filename = re.sub(' ', '_', query)
    filename = filename + '.csv'
    dirpath = mkdir_fromsearch(query)
    filepath = os.path.join(dirpath, filename)
    df.to_csv(filepath)


def scrapecomments(path, videoids: list, query: str, limit: int):
    """Download youtube comments
    Scrapes comments for each video id in the list
    """
    for vidid in tqdm(videoids):
        download_comments(path, vidid, limit)


def download_fromsearch(
    query, maxresults=10, commentcount=None,
    download=False, limit=None,
    to_csv=False
):
    """Youtube Comments Downloader
    Downloads n or all comments from a video id

    PARAMETERS
    ----------
    `query` : (str)
        A string used to find video ids by matching query keywords.

    `maxresults` : (int)
        Number of maximum results (matching videos) to get for a given query.

    `commentcount` : (int)
        Limits the videos that should be downloaded by N comments. For example,
        setting commentcount=1000, will download from videos with 1000 comments
        and greater. If None, then any video regardles of N comments for any ID

    `download` : (bool)
        If set to False, no comments will be downloaded.

    `limit` : (int)
        Limits how many comments should be downloaded per video id.

    `to_csv` :
        Save search results from the Youtube Data API to a csv file.
    """
    response = get_video_content(query, maxresults)
    df = pd.DataFrame.from_dict(response, orient="columns")
    df['commentCount'] = df['commentCount'].astype(int)
    # filter df w/videos containing 2000 comments or more
    df = df[df['commentCount'] > commentcount]
    videoids = df['vidId'].values.tolist()

    if download:
        scrapecomments(videoids, query, limit)
    if to_csv:
        save_tocsv(df, query)
    elif not (download and to_csv):
        pprint(response)


if __name__ == '__main__':

    QSEARCH = 'cnn china trump'
    download_fromsearch(QSEARCH, maxresults=10,
                        commentcount=2000, download=True, limit=None,
                        to_csv=True
                        )
