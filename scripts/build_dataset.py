import os
import re

import pandas as pd
from glob import glob

from vuepoint.pipeline.textstatistics import textmetrics


def assing_videoid_labels(filename: str):
    filename = re.sub('downloads/', '', str(filename))
    filename = re.sub('.json', '', filename)
    search_query, video_id = filename.split('/')
    return search_query, video_id


def save_dataset_to_directory(df: object, dirname: str, filename: str):
    if not os.path.exists(dirname):
        dirpath = os.makedirs(dirname)
    csv_filename = filename + '.csv'
    save_path = os.path.join(dirpath, csv_filename)
    df.to_csv(save_path)


if __name__ == "__main__":

    dataset_filename = input('\nenter name (only) of folder to preprocess: ')
    prodigy_dirname = input('\nenter name of the folder to save in prodigy: ')
    DATASET_PATH = str(f'downloads/{dataset_filename}/*.json')
    PRODIGY_DIRPATH = str(f'prodigy/working_datasets/{prodigy_dirname}')

    joined_datasets = []
    for dataset in glob(DATASET_PATH):
        df = pd.read_json(dataset, encoding='utf-8', lines=True)
        search_query, video_id = assing_videoid_labels(dataset)
        df['search_query'] = search_query
        df['video_id'] = video_id
        joined_datasets.append(df)

datasets = pd.concat(joined_datasets, ignore_index=True)
datasets = examinetext(datasets, 'text', 9)
save_dataset_to_directory(datasets, PRODIGY_DIRPATH,
                          filename=dataset_filename)

print("done saving dataset to prodigy directory!")
