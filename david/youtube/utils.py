import os

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


REGX_MATCH_SPECIAL_TAGS = {
    'match_titles': "(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'match_quotes': '"(?:\\.|(\\")|[^""\n])*"',
    'match_times': '([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'match_youtubeurl': '(?:https?:\\/\\/)?(?:(?:(?:www\\.?)?youtube\\.com(?:\\/(?:(?:watch\\?.*?(v=[^&\\s]+).*)|(?:v(\\/.*))|(channel\\/.+)|(?:user\\/(.+))|(?:results\\?(search_query=.+))))?)|(?:youtu\\.be(\\/.*)?))',
    'trim_whitespaces': '(?:\\s)\\s")'}

REGX_MATCH_TAGS = {
    'titles': r"(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'quotes': r'"(?:\\.|(\\")|[^""\n])*"',
    'times': r'([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'username_v1': r'\B(\@[a-zA-Z_0-9]+\b)(?!;)',
    'username_v2': r'(\@[a-zA-Z0-9_%]*)'}

REGX_MATCH_URLS = {
    'videoId': r'v=([a-zA-Z0-9\_\-]+)&?',
    'vid_url1': r'youtube.[a-z]+/[a-z\?\&]*v[/|=](\w+)',
    'any_vidUrl': r'(?:https?:\/\/)?(?:www\.)?youtu(.be\/|be\.com\/watch\?v=)(.{8,})',
    'vid_url2': r'(((\?v=)|(\/embed\/)|(youtu.be\/)|(\/v\/)|(\/a\/u\/1\/))(.+?){11})'}


CATEGORY_IDS = {
    'film.animation': '1',
    'autos.vehicles': '2',
    'music': '10',
    'pets.animals': '15',
    'sports': '17',
    'short.movies': '18',
    'travel.events': '19',
    'gaming': '20',
    'videoblogging': '21',
    'people.blogs': '22',
    'comedy': '34',
    'entertainment': '24',
    'news.politics': '25',
    'howto.style': '26',
    'education': '27',
    'science.technology': '28',
    'nonprofits.activism': '29',
    'movies': '30',
    'anime.animation': '31',
    'action.adventure': '32',
    'classics': '33',
    'documentary': '35',
    'drama': '36',
    'family': '37',
    'foreign': '38',
    'horror': '39',
    'scifi.fantasy': '40',
    'thriller': '41',
    'shorts': '42',
    'shows': '43',
    'trailers': '44',
}


class YTRegexMatchers:
    SPECIAL_TAGS = REGX_MATCH_SPECIAL_TAGS
    TAGS = REGX_MATCH_TAGS
    URLS = REGX_MATCH_URLS


class YTSpecialKeys:
    API = API
    CHANNEL_KEYS = CHANNEL_KEYS
    CATEGORY_IDS = CATEGORY_IDS