import os
import re

import pandas as pd
from tqdm import tqdm
from pprint import pprint

from vuepoint.youtube.search import get_video_content
from vuepoint.youtube.scraper import download_comments


def make_videoid_directory(query):
    """Names and creates a unique directory per search query
    """
    query_dirname = re.sub(' ', '_', query)
    query_dirpath = os.path.join('downloads', query_dirname)
    return query_dirpath


def search_query_to_csv(df, query):
    """Saves the CSV file to its parent directory.
    The file is named in relation to the a search query
    """
    csv_filename = re.sub(' ', '_', query)
    csv_filename = csv_filename + '.csv'
    query_dirpath = make_videoid_directory(query)
    csv_filepath = os.path.join(query_dirpath, csv_filename)
    df.to_csv(csv_filepath)


# NOTE: add the json file naming code to the actual downloader
# function and not inside the download_comments_from_videoids function


def download_comments_from_videoids(ids_list, query, limit):
    """Downloads youtube comments from a custom WebScraping script.
    Scrapes comments from each video id in the list.
    """
    query_dirpath = make_videoid_directory(query)
    for vid_id in tqdm(ids_list):
        vid_filename = ''.join(vid_id + '.json')
        download_comments(
            youtube_id=vid_id,
            file_name=vid_filename,
            limit=limit,
            dir_name=query_dirpath)


def build_corpus_from_query_results(
    query='',
    max_results=10,
    min_comments=None,
    download=True,
    comment_limit=None,
    to_csv=False
):
    """Downloads Youtube comments by Video ID from a given search query

    PARAMS
    ------
    query : str
        A string used to find video ids by matching query keywords

    max_results : int
        Number of maximum results (matching videos) to get for a given query

    min_comments : int
        Limits the videos that should be downloaded by N comments. For example,
        setting min_comments=1000, will download from videos with 1000 comments
        and greater. If None, then any video regardles of N comments for any ID

    download : bool
        If set to False, no comments will be downloaded

    comment_limit : int
        Limits how many comments should be downloaded per video id

    to_csv :
        Save search results from the Youtube Data API to a csv file
    """
    query_response = get_video_content(query, max_results)
    df = pd.DataFrame.from_dict(query_response, orient="columns")
    df['commentCount'] = df['commentCount'].astype(int)
    # get only videos with comment len > 2000
    df = df[df['commentCount'] > min_comments]
    # save results to the downloads directory
    video_id_list = df['vidId'].values.tolist()

    if download:
        download_comments_from_videoids(ids_list=video_id_list,
                                        query=query,
                                        limit=comment_limit)
    if to_csv:
        search_query_to_csv(df, query)
    if not (download and to_csv):
        pprint(query_response)


SEARCH_QUERY = 'company china trump world american'


def main():
    build_corpus_from_query_results(
        SEARCH_QUERY,
        max_results=20,
        min_comments=2000,
        download=True,
        comment_limit=None,
        to_csv=True)


if __name__ == '__main__':
    main()
