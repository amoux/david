from googleapiclient.discovery import build

from .utils import YtApiKeys


def yt_search(q: str, max_results=10):
    '''Returns a list of matching search results.
    '''
    youtube = YtApiKeys()
    resource = build(
        youtube.api.service,
        youtube.api.version,
        developerKey=youtube.api.key
    )
    search = resource.item().list(
        q=q,
        part='id, snippet',
        maxResults=max_results,
        order=youtube.stat.views
    ).execute()
    return search


def yt_channel(q: str, max_results=10):
    '''Youtube video content from a search query.
    (Youtube Data API). Returns a list of matching videos,
    channels matching the given a item query.

    PARAMETERS
    ----------

    `q` : (str)
    The item query (text) to item for videos on youtube,
    which influences the video_response based on the keywords given
    to the parameter.

    `max_results` : (int)
    Number of results to retrive for the given item query.
    '''
    youtube = YtApiKeys()
    resource = build(
        youtube.api.service,
        youtube.api.version,
        developerKey=youtube.api.key
    )
    search = yt_search(q, max_results)

    results = []
    for item in search.get('items', []):

        if item['id']['kind'] == 'youtube#video':
            temp = {}
            temp['title'] = item['snippet']['title']
            temp['vidId'] = item['id']['videoId']

            videos = resource.videos().list(
                part='statistics, snippet',
                id=item['id']['videoId']).execute()

            items = videos['items'][0]['snippet']
            for content in youtube.content._fields:
                try:
                    temp[content] = items[content]
                except KeyError:
                    temp[content] = 'xxNoneFoundxx'

            items = videos['items'][0]['statistics']
            for stat in youtube.stat._fields:
                try:
                    temp[stat] = items[stat]
                except KeyError:
                    temp[stat] = 'xxNoneFoundxx'

            results.append(temp)
    return results
