import numbers
import os

import numpy

from ..utils import get_data_home

JSON_DATASETS = {
    "politics": [
        "china_prepared_Trump",
        "cnn_china_trump",
        "politics_news_trump"
    ],
    "trends": [
        "best_smart_phones_of_2019",
        "worst_car_trends",
        "worst_smart_phones_of_2019"
    ],
    "howto": [
        "how_to_improve_yourself",
        "how_to_program_python",
        "getting_more_views_on_youtube"
    ],
    "reviews": [
        "thinkpad_review",
        "ferrari_sf90"
    ],
    "events": [
        "electric_cars_are_the_future",
        "is_this_the_end_of_huawei"
    ],
    "blogs": ["reading_youtube_comments"]
}


class Bunch(dict):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass


def check_random_state(seed):
    if seed is None or seed is numpy.random:
        return numpy.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return numpy.random.RandomState(seed)
    if isinstance(seed, numpy.random.RandomState):
        return seed
    raise ValueError(
        f"{seed} cannot be used to seed a \
        numpy.random.RandomState instance.")


def load_files(container_path, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error="strict", random_state=0):
    """Load text files with categories as subfolder names."""
    target = list()
    target_names = list()
    filenames = list()

    folders = [f for f in sorted(os.listdir(container_path))
               if os.path.isdir(os.path.join(container_path, f))]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = os.path.join(container_path, folder)
        documents = [os.path.join(folder_path, d)
                     for d in sorted(os.listdir(folder_path))]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    filenames = numpy.array(filenames)
    target = numpy.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = numpy.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = list()
        for filename in filenames:
            with open(filename, "rb") as f:
                data.append(f.read())

        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]

        return Bunch(data=data,
                     filenames=filenames,
                     target_names=target_names,
                     target=target,
                     DESCR=description)

    return Bunch(filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)


class JsonlYTDatasets:
    CATEGORIES = JSON_DATASETS

    def __init__(self, files_dirpath=None):
        self.files_dirpath = files_dirpath
        if not self.files_dirpath:
            david_data_path = get_data_home()
            self.files_dirpath = os.path.join(david_data_path, "jsonl")

    def file_paths(self, category=None, load_content=False):
        if not category or category not in self.CATEGORIES.keys():
            raise Exception("Choose a category to load: {}".format(
                self.CATEGORIES.keys()))

        return load_files(
            self.files_dirpath,
            categories=self.CATEGORIES[category],
            load_content=load_content)
