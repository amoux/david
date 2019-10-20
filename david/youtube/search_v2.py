from typing import Dict, Iterable, List

import requests
from isodate import parse_duration

from ..configs.youtube import API as _API


def filter_by_comment_counts(dict_list: list, min_comments: int):
    filtered = []
    for key in dict_list:
        if int(key['comments']) >= min_comments:
            filtered.append(key)
    return filtered


def request(url, params, req_key='items') -> Dict:
    req = requests.get(url, params=params)
    return req.json()[req_key]


def search(q: str, max_results: int) -> List:
    """Returns a list of video ids matching the query."""
    search_params = {
        'key': _API['api_key'],
        'q': q, 'part': 'snippet',
        'maxResults': max_results,
        'type': 'video'}

    video_ids = list()
    for result in request(_API['search_url'], search_params):
        video_ids.append(result['id']['videoId'])
    return video_ids


def video(q: str, max_results: int) -> Iterable[List[Dict]]:
    video_params = {
        'key': _API['api_key'],
        # joins (str) from the list of video ids from the response.
        'id': ','.join(search(q, max_results)),
        'part': 'snippet, contentDetails, statistics',
        'maxResults': max_results}

    videos = list()
    for result in request(_API['video_url'], video_params):
        video_data = {
            'id': result['id'],
            'url': f'https://www.youtube.com/watch?v={result["id"]}',
            'thumbnail': result['snippet']['thumbnails']['high']['url'],
            'duration': int(parse_duration(
                result['contentDetails']['duration']).total_seconds()//60),
            'title': result['snippet']['title'],
            'comments': result['statistics']['commentCount']}

        videos.append(video_data)
    return videos
