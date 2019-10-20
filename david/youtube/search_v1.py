from typing import Dict, Iterable, List

from googleapiclient import discovery

from ..configs.youtube import API as _API
from ..configs.youtube import CHANNEL_KEYS as _CHANNEL_KEYS


def search(q: str, max_results: int = 10) -> Dict:
    """Returns a list of matching search results."""
    service = discovery.build(
        serviceName=_API['service'],
        version=_API['api_version'],
        developerKey=_API['api_key'])
    search_service = service.search()

    return search_service.list(
        q=q, part='id, snippet', maxResults=max_results,
        order=_CHANNEL_KEYS['order']).execute()


def video(q: str, max_results: int = 10) -> Iterable[List[Dict]]:
    """Youtube video content from a search query.

    Returns a list of matching videos, channels matching
    the given a item query.

    Parameters:
    ----------
    q : (str)
        The item query (text) to item for videos on youtube,
        which influences the video_response based on the keywords given
        to the parameter.
    max_results : (int)
        Number of results to retrive for the given item query.
    """
    service = discovery.build(
        serviceName=_API['service'],
        version=_API['api_version'],
        developerKey=_API['api_key'])

    video_service = service.videos()
    search_service = search(q, max_results)

    results, temp = list(), dict()
    for item in search_service.get('items', []):
        if item['id']['kind'] == 'youtube#video':
            temp['title'] = item['snippet']['title']
            temp['vidId'] = item['id']['videoId']
            videos = video_service.list(
                part='statistics, snippet',
                id=item['id']['videoId']).execute()

            for content in _CHANNEL_KEYS['content']:
                try:
                    temp[content] = videos['items'][0]['snippet'][content]
                except KeyError:
                    temp[content] = 'xxNoneFoundxx'

            for stats in _CHANNEL_KEYS['stats']:
                try:
                    temp[stats] = videos['items'][0]['statistics'][stats]
                except KeyError:
                    temp[stats] = 'xxNoneFoundxx'

            results.append(temp)
    return results
