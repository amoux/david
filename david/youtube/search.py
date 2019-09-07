from typing import Dict, Iterable, List

from googleapiclient import discovery

from david.config import YoutubeConfig

youtube = YoutubeConfig()


def yt_search(q: str, max_results: int = 10) -> Dict:
    '''Returns a list of matching search results.
    '''
    Discovery = discovery.build(
        serviceName=youtube.api.service,
        version=youtube.api.version,
        developerKey=youtube.api.key
    )
    searchResource = Discovery.search()
    search = searchResource.list(
        q=q,
        part='id, snippet',
        maxResults=max_results,
        order=youtube.stat.views
    ).execute()
    return search


def yt_video(q: str, max_results: int = 10) -> Iterable[List[Dict]]:
    '''Youtube video content from a search query.
    (Youtube Data API). Returns a list of matching videos,
    channels matching the given a item query.

    Parameters:
    ----------

    `q` : (type=str)
        The item query (text) to item for videos on youtube,
        which influences the video_response based on the keywords given
        to the parameter.

    `max_results` : (type=int)
        Number of results to retrive for the given item query.

    '''
    Discovery = discovery.build(
        serviceName=youtube.api.service,
        version=youtube.api.version,
        developerKey=youtube.api.key
    )
    videoResource = Discovery.videos()
    search = yt_search(q, max_results)

    results = []
    for item in search.get('items', []):
        if (item['id']['kind'] == 'youtube#video'):

            temp = {}
            temp['title'] = item['snippet']['title']
            temp['vidId'] = item['id']['videoId']

            videos = videoResource.list(
                part='statistics, snippet',
                id=item['id']['videoId']
            ).execute()

            items = videos['items'][0]['snippet']
            for content in youtube.content:
                try:
                    temp[content] = items[content]
                except KeyError:
                    temp[content] = 'xxNoneFoundxx'

            items = videos['items'][0]['statistics']
            for stat in youtube.stat:
                try:
                    temp[stat] = items[stat]
                except KeyError:
                    temp[stat] = 'xxNoneFoundxx'

            results.append(temp)
    return results
