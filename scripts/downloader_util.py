import os
import re
from pprint import pprint

import pandas as pd
from tqdm import tqdm

from david.youtube import scrape_comments_to_json, search_v1


def create_directory(query: str, save_path: str):
    """Creates directory names based on the query searched."""
    dir_name = re.sub(' ', '_', query)
    return os.path.join(save_path, dir_name)


def save_corpus(df: object, query: str,
                save_path: str, join_by='_', ftype='.csv'):
    """Saves the corpus file to its parent directory."""
    file_name = re.sub(' ', join_by, query)
    file_path = os.path.join(
        create_directory(query, save_path), f'{file_name}{ftype}')
    df.to_csv(file_path)


def download_comments(videoids: list, file_path: str, limit: int):
    for vid in tqdm(videoids):
        scrape_comments_to_json(
            video_id=vid, dirpath=file_path, limit=limit)


def build_corpus(
        query,
        max_results=10,
        min_comments=None,
        limit=None,
        download=False,
        save_path='downloads',
        to_csv=False,
        join_by='_'):
    """Youtube Comments Courpus Builder.

    Downloads comments from a youtube video id and saves
    the results to a csv file.

    Parameters:
    ----------
    query : (str)
        A string used to find video ids by
        matching query keywords.
    max_results : (int)
        Number of maximum results (matching videos)
        to get for a given query.
    min_comments : (int)
        Min comments a video id requires, to be
        worthy of downloading.
    download : (bool)
        If set to False, no comments will be downloaded.
    limit : (int)
        Limits how many comments should be downloaded per
        video id.
    to_csv :
        Save search results from the Youtube Data API to
        a csv file.
    """
    query_response = search_v1(query, max_results)
    df = pd.DataFrame.from_dict(query_response, orient="columns")
    df['commentCount'] = df['commentCount'].astype(int)
    df = df.loc[df['commentCount'] > min_comments]
    video_ids = df['vidId'].values.tolist()

    if download:
        download_comments(video_ids, save_path, limit)
    if to_csv:
        save_corpus(df, query, save_path, join_by)
    elif not download and to_csv:
        pprint(query_response)


if __name__ == '__main__':

    query = 'python is the future'
    build_corpus(
        query,
        max_results=10,
        min_comments=2000,
        download=True,
        limit=None,
        to_csv=True)
