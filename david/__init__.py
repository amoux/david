from collections import namedtuple as _ntuple


_REGX_POINTERS = {
    'match_titles': "(-?([A-Z].\\s)?([A-Z][a-z]+)\\s?)+([A-Z]'([A-Z][a-z]+))?",
    'match_quotes': '"(?:\\.|(\\")|[^""\n])*"',
    'match_times': '([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?',
    'match_youtubeurl': '(?:https?:\\/\\/)?(?:(?:(?:www\\.?)?youtube\\.com(?:\\/(?:(?:watch\\?.*?(v=[^&\\s]+).*)|(?:v(\\/.*))|(channel\\/.+)|(?:user\\/(.+))|(?:results\\?(search_query=.+))))?)|(?:youtu\\.be(\\/.*)?))',
    'trim_whitespaces': '(?:\\s)\\s")'
}

_DATASET_PATHS = {
    'creator_videoid_info': "~/Visual-Studio/Vuepoint-Analytics/Api-Project/david.Apiv1/david/datasets/ycd_csv/ycc_authors_info.csv",
    'ycd_web_tensorboard': "~/Visual-Studio/Vuepoint-Analytics/Api-Project/david.Apiv1/david/datasets/ycd_csv/ycd_web_tensorboard.csv",
    'ycd_web_md': "~/Visual-Studio/Vuepoint-Analytics/Api-Project/david.Apiv1/david/datasets/ycd_csv/ycc_web_lg.csv",
    'ycd_web_lg': "~/Visual-Studio/Vuepoint-Analytics/Api-Project/david.Apiv1/david/datasets/ycd_csv/ycc_dataset_lg_ready.csv"
}


def load_regex():
    '''Loads special regex collection keys.
    '''
    named = _ntuple('Regex', _REGX_POINTERS.keys())
    return named(*_REGX_POINTERS.values())


def load_datasets():
    '''Loads collection of datasets.
    '''
    named = _ntuple('YCD', _DATASET_PATHS.keys())
    return named(*_DATASET_PATHS.values())
