import collections
import os
import urllib
import zipfile

_MOVIELENS_URLS = {
    "ml-full": "http://files.grouplens.org/datasets/movielens/ml-latest.zip",
    "ml-small": "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-youtube": "http://files.grouplens.org/datasets/movielens/ml-20m-youtube.zip",
}

MOVIELENS_DATASETS = {
    "lg": [_MOVIELENS_URLS["ml-full"]],
    "sm": [_MOVIELENS_URLS["ml-small"]],
    "yt": [_MOVIELENS_URLS["ml-youtube"]],
    "all": _MOVIELENS_URLS.values()}


def download_movielens(dataset: str, savepath="data"):
    """Downloads and configures MovieLens datasets.

    About MovieLens: The data are contained in the files links.csv, movies.csv,
    ratings.csv and tags.csv.

    Parameters:
    ----------

    `dataset` : (str, choices: ['sm', 'lg', 'yt', 'all'])
        Choices for the MovieLens datasets: `sm` or `lg`. To download the
        MovieLens youtube trailers dataset: `yt`. To download all datasets
        `all`.

    `savepath` : (str, default="data")
        The directory where the datasets will be downloaded and configured.
        If the downloaded the datasets (e.g, dataset="all") exists already,
        you can use this method and pass the same argument "all" to reload
        the dataset contents into memory. (It wont re-download if exists).

    """
    extlen = len(".zip")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    dataset_fpaths = collections.defaultdict(list)
    for dataset in MOVIELENS_DATASETS[dataset]:
        filename = dataset[dataset.rfind('ml'):]
        filepath = os.path.join(savepath, filename)
        if not os.path.exists(filepath[:-extlen]):
            download = urllib.request.urlretrieve(dataset, filepath)
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(savepath)
        dataset_fpaths[filename[:-extlen]] = {
            fn[:-extlen]: os.path.join(filepath[:-extlen], fn)
            for fn in os.listdir(filepath[:-extlen])}
    return dataset_fpaths
