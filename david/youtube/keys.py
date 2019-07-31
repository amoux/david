# CONFIGURATIONS FOR YOUTUBE API V3

from functools import namedtuple
from os import environ


# DEVELOPER

_YOUTUBE_API_V3KEY = environ.get('YOUTUBE_API_V3KEY')
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'

# CHANNEL

CHANNEL_CONTENT_KEYS = [
    'published',
    'channel_id',
    'description',
    'title',
    'tags',
    'category_id',
    'live_content',
    'default_lang'
]

CHANNEL_STAT_KEYS = ['favorites', 'views', 'likes', 'dislikes', 'comments']


class ChannelKeys:
    def __init__(self):
        """Youtube V3 Channel Keys
        """
        self.CHANNEL_CONTENT_KEYS = CHANNEL_CONTENT_KEYS
        self.CHANNEL_STAT_KEYS = CHANNEL_STAT_KEYS

        self.content = self._get_youtube_keys("content")
        self.stat = self._get_youtube_keys("stats")
        self.dev = self._get_youtube_keys("dev")

    def _get_youtube_keys(self, list_type: str):
        if list_type == "content":
            content = namedtuple('ChannelContent', self.CHANNEL_CONTENT_KEYS)
            return content(
                'publishedAt', 'channelId',
                'description', 'channelTitle',
                'tags', 'categoryId',
                'liveBroadcastContent', 'defaultLanguage'
            )
        if list_type == "stats":
            stats = namedtuple('ChannelStats', self.CHANNEL_STAT_KEYS)
            return stats('favoriteCount', 'viewCount', 'likeCount',
                         'dislikeCount', 'commentCount')

        if list_type == "dev":
            dev = namedtuple(
                'DevSettings', ['key', 'service', 'version'])
            return dev(_YOUTUBE_API_V3KEY, API_SERVICE_NAME, API_VERSION)


# FULL KEYS THAT CAN BE SEARCHED
# search_list = ['publishedAt', 'channelId', 'description', 'channelTitle',
# 'tags', 'categoryId', 'liveBroadcastContent', 'defaultLanguage']

# stats_list = ['favoriteCount','viewCount','likeCount',
# 'dislikeCount','commentCount']
