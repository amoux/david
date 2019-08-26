from functools import namedtuple

REGEX_DICT = {
    'match_titles': "(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'match_quotes': '"(?:\\.|(\\")|[^""\n])*"',
    'match_times': '([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'match_youtubeurl': '(?:https?:\\/\\/)?(?:(?:(?:www\\.?)?youtube\\.com(?:\\/(?:(?:watch\\?.*?(v=[^&\\s]+).*)|(?:v(\\/.*))|(channel\\/.+)|(?:user\\/(.+))|(?:results\\?(search_query=.+))))?)|(?:youtu\\.be(\\/.*)?))',
    'trim_whitespaces': '(?:\\s)\\s")'
}


def pointer(name: str, params: dict):
    '''Returns a dictionary-like object from its
    key, value pairs.
    '''
    name = namedtuple(name, params.keys())
    return name(*params.values())
