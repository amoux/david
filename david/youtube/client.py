import json
import urllib

from googleapiclient import discovery

from .utils import API, CHANNEL_KEYS


def yt_query_search(q: str, max_results: int = 10):
    """Returns a list of matching search results."""
    service = discovery.build(
        serviceName=API['service'],
        version=API['api_version'],
        developerKey=API['api_key'])
    search_service = service.search()

    return search_service.list(
        q=q, part='id, snippet', maxResults=max_results,
        order=CHANNEL_KEYS['order']).execute()


def yt_query_video_content(q: str, max_results: int = 10):
    """Youtube video content from a search query.

    Returns a list of matching videos matching the given a item query.

    Args:
        `q` : (str) The item query (text) to item for videos on youtube,
        which influences the video_response based on the keywords given
        to the parameter.
        `max_results` : (int) Number of results to retrive for the given query.
    """
    service = discovery.build(
        serviceName=API['service'],
        version=API['api_version'],
        developerKey=API['api_key'])

    video_service = service.videos()
    search_service = yt_query_search(q, max_results)

    results, temp = list(), dict()
    for item in search_service.get('items', []):
        if item['id']['kind'] == 'youtube#video':
            temp['title'] = item['snippet']['title']
            temp['vidId'] = item['id']['videoId']
            videos = video_service.list(
                part='statistics, snippet',
                id=item['id']['videoId']).execute()

            for content in CHANNEL_KEYS['content']:
                try:
                    temp[content] = videos['items'][0]['snippet'][content]
                except KeyError:
                    temp[content] = 'xxNoneFoundxx'

            for stats in CHANNEL_KEYS['stats']:
                try:
                    temp[stats] = videos['items'][0]['statistics'][stats]
                except KeyError:
                    temp[stats] = 'xxNoneFoundxx'

            results.append(temp)
    return results


def yt_query_videoids(channel_id: str, max_results=25):
    search_url = 'https://www.googleapis.com/youtube/v3/search?'
    url_params = 'key={}&channelId={}&part=snippet,id&order=date&maxResults={}'
    first_url = search_url + url_params.format(
        API['api_key'], channel_id, max_results)

    video_ids = list()
    url = first_url
    while True:
        req_url = urllib.request.urlopen(url)
        response = json.load(req_url)
        for i in response['items']:
            if i['id']['kind'] == "youtube#video":
                video_ids.append(i['id']['videoId'])
        try:
            next_page_token = response['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        except ValueError:
            break
    return video_ids
