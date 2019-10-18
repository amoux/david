from typing import Dict, Iterable, List

from googleapiclient import discovery

from ..config import YoutubeConfig

_youtube = YoutubeConfig()


def _search(q: str, max_results: int = 10) -> Dict:
    """Returns a list of matching search results."""
    Discovery = discovery.build(
        serviceName=_youtube.api.service,
        version=_youtube.api.version,
        developerKey=_youtube.api.key)
    searchResource = Discovery.search()
    search = searchResource.list(
        q=q,
        part='id, snippet',
        maxResults=max_results,
        order=_youtube.stat.views).execute()
    return search


def _video(q: str, max_results: int = 10) -> Iterable[List[Dict]]:
    """Youtube video content from a search query.

    Returns a list of matching videos, channels matching
    the given a item query.

    Parameters:
    ----------
    q : (type=str)
        The item query (text) to item for videos on youtube,
        which influences the video_response based on the keywords given
        to the parameter.
    max_results : (type=int)
        Number of results to retrive for the given item query.
    """
    Discovery = discovery.build(
        serviceName=_youtube.api.service,
        version=_youtube.api.version,
        developerKey=_youtube.api.key)
    videoResource = Discovery.videos()
    search = _search(q, max_results)

    results = []
    for item in search.get('items', []):
        if (item['id']['kind'] == 'youtube#video'):
            temp = {}
            temp['title'] = item['snippet']['title']
            temp['vidId'] = item['id']['videoId']

            videos = videoResource.list(
                part='statistics, snippet',
                id=item['id']['videoId']).execute()
            items = videos['items'][0]['snippet']

            for content in _youtube.content:
                try:
                    temp[content] = items[content]
                except KeyError:
                    temp[content] = 'xxNoneFoundxx'

            items = videos['items'][0]['statistics']
            for stat in _youtube.stat:
                try:
                    temp[stat] = items[stat]
                except KeyError:
                    temp[stat] = 'xxNoneFoundxx'

            results.append(temp)
    return results
