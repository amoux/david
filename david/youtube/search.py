from googleapiclient.discovery import build
from .keys import ChannelKeys


def _records_fromterm(Qsearch, max_results=10):
    """Calls the `search().list([_Qsearch_])` method to retrieve
    Return list of matching records up to `max_results`.
    """
    CK = ChannelKeys()
    DEV = CK.dev
    STAT = CK.stat

    youtube = build(DEV.service, DEV.version, developerKey=DEV.key)
    search_response = youtube.search().list(
        q=Qsearch,
        part='id,snippet',
        maxResults=max_results,
        order=STAT.views
    ).execute()
    return search_response


def get_video_content(Qsearch, max_results=10):
    """Get Video-Content Response from a Search Query.
    (Youtube Data API). Returns a list of matching videos,
    channels matching the given `Qsearch`.

    PARAMETERS
    ----------
    `Qsearch` : (str)
        The search query (text) to search for videos on youtube,
        which influences the response based on the keywords given
        to the parameter.

    `max_results` : (int)
        Number of results to retrive for the given search query.

    NOTE: Additional parameters can be used;

    `order="viewCount"`, `token=None`, `location=None`, `location_radius=None`
    """
    CK = ChannelKeys()
    DEV = CK.dev

    youtube = build(DEV.service, DEV.version, developerKey=DEV.key)
    search_response = _records_fromterm(Qsearch, max_results)

    search_results = []
    for search in search_response.get("items", []):
        if search["id"]["kind"] == 'youtube#video':
            # available from initial search
            tempdict = {}
            tempdict['title'] = search['snippet']['title']
            tempdict['vidId'] = search['id']['videoId']
            # secondary call to find statistics
            # results for individual videos
            response = youtube.videos().list(
                part='statistics,snippet', id=search['id']['videoId']
                ).execute()
            response_stats = response['items'][0]['statistics']
            response_snippet = response['items'][0]['snippet']
            for key in CK.CHANNEL_CONTENT_KEYS:
                try:
                    tempdict[key] = response_snippet[key]
                except KeyError:
                    # not stored if not present
                    tempdict[key] = 'xxNoneFoundxx'
            for key in CK.CHANNEL_STAT_KEYS:
                try:
                    tempdict[key] = response_stats[key]
                except KeyError:
                    # not stored if not present
                    tempdict[key] = 'xxNoneFoundxx'
            # add back to main list
            search_results.append(tempdict)
    return search_results
