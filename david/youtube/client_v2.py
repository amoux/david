import requests
from isodate import parse_duration

from .utils import API


def request(url, params, req_key='items'):
    req = requests.get(url, params=params)
    return req.json()[req_key]


def search(q: str, max_results: int):
    """Returns a list of video ids matching the query."""
    search_params = {
        'key': API['api_key'],
        'q': q, 'part': 'snippet',
        'maxResults': max_results,
        'type': 'video'}

    video_ids = list()
    for result in request(API['search_url'], search_params):
        video_ids.append(result['id']['videoId'])
    return video_ids


def video(q: str, max_results: int):
    video_params = {
        'key': API['api_key'],
        # joins (str) from the list of video ids from the response.
        'id': ','.join(search(q, max_results)),
        'part': 'snippet, contentDetails, statistics',
        'maxResults': max_results}

    videos = list()
    for result in request(API['video_url'], video_params):
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
