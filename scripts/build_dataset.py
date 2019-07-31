import os
import re

import pandas as pd
from glob import glob

from david.pipeline.feature_engineering import textmetrics


def get_videolabels(fname: str, path='downloads/', ftype='.json'):
    # this is an useless function. it only worked for one type of thing.
    fname = fname.replace(path, '').replace(ftype, '')
    query, videoid = fname.split('/')
    return query, videoid


def dataset_tosavedir(df: object, fname: str, dirname: str):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif os.path.exists(dirname):
        fname = (fname + '.csv')
        fullpath = os.path.join(dirname, fname)
    df.to_csv(fullpath)


if __name__ == "__main__":

    fname = input('\nenter name (only) of folder to preprocess: ')
    dirname = input('\nenter name of the folder to save in prodigy: ')
    DATASET_PATH = f'downloads/{fname}/*.json'
    PRODIGY_DIRPATH = f'prodigy/working_datasets/{dirname}'

    joined_datasets = []
    for dataset in glob(DATASET_PATH):
        df = pd.read_json(dataset, encoding='utf-8', lines=True)
        query, videoid = get_videolabels(dataset)
        df['query'] = query
        df['videoid'] = videoid
        joined_datasets.append(df)

datasets = pd.concat(joined_datasets, ignore_index=True)
datasets = textmetrics(datasets, 'text', 9)
dataset_tosavedir(datasets, PRODIGY_DIRPATH, fname)
print("done saving dataset to prodigy directory!")
