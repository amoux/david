import os
import re
from glob import glob

import pandas as pd

from david.pipeline import TextMetrics


def get_videolabels(fn: str, fp='downloads/', ftype='.json'):
    # this is an useless function. it only worked for one type of thing.
    fn = fn.replace(fp, '').replace(ftype, '')
    query, videoid = fn.split('/')
    return (query, videoid)


def dataset_tosavedir(df: object, fn: str, fp: str):
    if not os.path.exists(fp):
        os.makedirs(fp)
    elif os.path.exists(fp):
        fn = (fn + '.csv')
        fullpath = os.path.join(fp, fn)
    df.to_csv(fullpath)

metric = TextMetrics()

if __name__ == "__main__":

    fn = input('\nenter name (only) of folder to preprocess: ')
    fp = input('\nenter name of the folder to save in prodigy: ')

    DATASET_PATH = f'downloads/{fn}/*.json'
    PRODIGY_DIRPATH = f'prodigy/working_datasets/{fp}'

    joined_datasets = []
    for dataset in glob(DATASET_PATH):
        df = pd.read_json(dataset, encoding='utf-8', lines=True)
        query, videoid = get_videolabels(dataset)
        df['query'] = query
        df['videoid'] = videoid
        joined_datasets.append(df)

datasets = pd.concat(joined_datasets, ignore_index=True)
datasets = TextMetrics(datasets, 'text', 9)
dataset_tosavedir(datasets, PRODIGY_DIRPATH, fn)
print("done saving dataset to prodigy directory!")
