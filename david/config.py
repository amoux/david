from os import environ

from david.utils import pointer

YT_API = {
    'key': environ.get('YOUTUBE_API_KEY'),
    'service': 'youtube',
    'version': 'v3',
    'search_url': 'https://www.googleapis.com/youtube/v3/search',
    'video_url': 'https://www.googleapis.com/youtube/v3/videos'
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


class YoutubeConfig:
    def __init__(self):
        self.api = pointer('API', YT_API)
        self.stat = pointer('Stats', CHANNEL_STATS_API)
        self.content = pointer('Content', CHANNEL_CONTENT_API)

    def __repr__(self):
        return f'YoutubeConfig(\n{self.api}\n{self.stat}\n{self.content})'
