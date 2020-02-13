import os
from typing import IO, List, Tuple, Union

import pandas as pd

from ..io.utils import GoogleDriveDownloader
from ..text.utils import split_train_test
from ..utils import get_data_home

# collection of databases (datasets) available to download:
GDRIVE_SQLITE_DATABASES = {
    "v1": [{"name": "comments_v1.db", "id": "1n1dV2hYR8gQMt5S3x6d2yF_9Dz8tDrGk"}],
    "v2": [{"name": "comments_v2.db", "id": "1AvIGoL4ZXOOpGpI4oexV-k0a-KavYnCX"}],
    "unbox": [
        {"name": "lew_comments_unbox.db", "id": "1FDoLyeee1o8nFs-GjLiXEv--eb8oC3vF"},
        {"name": "lew_video_unbox.db", "id": "118tWFthG8VcGAK0N1wIbFVU-q0ZLsC5K"},
    ],
}


def download_sqlite_database(
    db_names: Tuple[str, ...], david_home: str = None, return_destination: bool = False
) -> IO:
    """Download the main datasets available from google drive into the david_home directory.

    db_names: A database file to download. Available keys ('v1', 'v2', 'unbox').
    david_home: A user's home direcory.
    return_destination: Return the destination path of the file downloaded (optional)
    """
    if not david_home:
        david_home = get_data_home()

    gdrive = GoogleDriveDownloader(dirpath=david_home)
    sqlite_files = {k: v for k, v in GDRIVE_SQLITE_DATABASES.items() if k in db_names}

    for key, value in sqlite_files.items():
        db_files = [(g["name"], g["id"]) for g in value]
        for db_name, db_id in db_files:
            gdrive.desc = f"Downloading {db_name} from server 📡"
            gdrive.filename = db_name
            gdrive.download(db_id)
    if return_destination:
        return gdrive._destination


class YTCommentsDataset:
    """Temporary `configuration` while prototyping."""

    VOCAB_PATH = "/home/ego/david_data/vocab/yt_web_md/"
    VOCAB_FILE = os.path.join(VOCAB_PATH, "vocab.txt")
    CSV_CORPUS = os.path.join(VOCAB_PATH, "corpus.csv")
    model_name = "yt-web-md"
    num_samples = 61478

    @staticmethod
    def load_dataset_as_df() -> Union[pd.DataFrame, None]:
        """Load the corpus and returns a `pandas.Dataframe` instance."""
        return pd.read_csv(YTCommentsDataset.CSV_CORPUS)

    @staticmethod
    def load_dataset_as_doc() -> List[str]:
        """Return an generator of iterable string sequences."""
        df = YTCommentsDataset.load_dataset_as_df()
        for sequence in df["text"].tolist():
            yield sequence

    @staticmethod
    def split_train_test(k=6000, subset=0.8) -> Tuple[List[str], List[str]]:
        """Randomly split into train and test iterables.

        `k`: The split decision value or items in the dataset to consider to subset.
           e.g, If you want to use `k=100` samples from `samples=1000` then -> 
           train and test sizes: `< Train[80], Test[20] >` are returned.

        Returns `Tuple[List[str], List[str]]`: Two iterables - train_doc is 8/10 of
            the total while the test_doc is 2/10 of the total.
        """
        dataset = YTCommentsDataset.load_dataset_as_doc()
        return split_train_test(list(dataset), k, subset=subset)
