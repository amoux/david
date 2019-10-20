import os

from ..utils._data_util import pointer

API = {
    'api_key': os.environ.get('YOUTUBE_API_KEY'),
    'client_id': os.environ.get('YOUTUBE_CLIENT_ID'),
    'client_secret': os.environ.get('YOUTUBE_CLIENT_SECRET'),
    'service': 'youtube',
    'api_version': 'v3',
    'search_url': 'https://www.googleapis.com/youtube/v3/search',
    'video_url': 'https://www.googleapis.com/youtube/v3/videos'
}

# these keys are used only with the search_v1 module.
# adding or removing any affects the search results from the api.
CHANNEL_KEYS = {
    # the order to retrive the search results:
    'order': 'viewCount',
    # keys for the statistic results to retrive.
    'stats': [
        'favoriteCount', 'viewCount', 'likeCount',
        'dislikeCount', 'commentCount'
    ],
    # keys for the content results to retrive.
    'content': [
        'publishedAt', 'channelId', 'description',
        'channelTitle', 'tags', 'categoryId',
        'liveBroadcastContent', 'defaultLanguage'
    ],
}
