from functools import namedtuple
from os import environ


YT_API = {
    'key': environ.get('YOUTUBE_API_KEY'),
    'service': 'youtube',
    'version': 'v3'
}

CHANNEL_CONTENT_API = {
    'published': 'publishedAt',
    'channel_id': 'channelId',
    'description': 'description',
    'title': 'channelTitle',
    'tags': 'tags',
    'category_id': 'categoryId',
    'live_content': 'liveBroadcastContent',
    'lang': 'defaultLanguage'
}

CHANNEL_STATS_API = {
    'favorites': 'favoriteCount',
    'views': 'viewCount',
    'likes': 'likeCount',
    'dislikes': 'dislikeCount',
    'comments': 'commentCount'
}


def _pointer(name: str, params: dict):
    name = namedtuple(name, params.keys())
    return name(*params.values())


class YtApiKeys:
    def __init__(self):
        self.api = YtApiKeys._get('api')
        self.stat = YtApiKeys._get('stats')
        self.content = YtApiKeys._get('content')

    @classmethod
    def _get(cls, key: str):
        if key == 'api':
            return _pointer('YtAPI', YT_API)
        if key == 'content':
            return _pointer('YtContent', CHANNEL_CONTENT_API)
        if key == 'stats':
            return _pointer('YtStats', CHANNEL_STATS_API)
