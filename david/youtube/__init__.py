
from ._scraper import download
from ._search_v1 import _search as search_v1
from ._search_v2 import YoutubeConfig
from ._search_v2 import _search as search_v2

YOUTUBE_CATEGORIES = {
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
