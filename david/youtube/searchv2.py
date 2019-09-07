import os
from os import environ

import requests
from isodate import parse_duration

from david.config import YoutubeConfig

youtube = YoutubeConfig()


def yt_request(url, params, req_key='items'):
    r = requests.get(url, params=params)
    return r.json()[req_key]


def yt_search(q: str, max_results: int):
    search_params = {
        'key': youtube.api.key,
        'q': q,
        'part': 'snippet',
        'maxResults': max_results,
        'type': 'video'
    }
    video_ids = []
    for result in yt_request(youtube.api.search_url, search_params):
        video_ids.append(result['id']['videoId'])
    return video_ids


def yt_video(q: str, max_results: int):
    video_params = {
        'key': youtube.api.key,
        'id': ','.join(yt_search(q, max_results)),
        'part': 'snippet, contentDetails, statistics',
        'maxResults': max_results
    }
    videos = []
    for result in yt_request(youtube.api.video_url, video_params):
        video_data = {
            'id': result['id'],
            'url': f'https://www.youtube.com/watch?v={result["id"]}',
            'thumbnail': result['snippet']['thumbnails']['high']['url'],
            'duration': int(parse_duration(
                result['contentDetails']['duration']).total_seconds()//60),
            'title': result['snippet']['title'],
            'comments': result['statistics']['commentCount']
        }
        videos.append(video_data)
    return videos


def filter_by_comments(dict_list: list, min_comments: int):
    filtered = []
    for key in dict_list:
        if int(key['comments']) >= min_comments:
            filtered.append(key)
    return filtered