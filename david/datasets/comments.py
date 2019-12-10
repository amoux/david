from typing import IO, Tuple

from ..io.utils import GoogleDriveDownloader
from ..utils import get_data_home

# collection of databases (datasets) available to download:
GDRIVE_SQLITE_DATABASES = {
    'v1': [
        {
            'name': 'comments_v1.db',
            'id': '1n1dV2hYR8gQMt5S3x6d2yF_9Dz8tDrGk'
        }
    ],
    'v2': [
        {
            'name': 'comments_v2.db',
            'id': '1AvIGoL4ZXOOpGpI4oexV-k0a-KavYnCX'
        }
    ],
    'unbox': [
        {
            'name': 'lew_comments_unbox.db',
            'id': '1FDoLyeee1o8nFs-GjLiXEv--eb8oC3vF'
        },
        {
            'name': 'lew_video_unbox.db',
            'id': '118tWFthG8VcGAK0N1wIbFVU-q0ZLsC5K'
        }
    ]
}


def download_sqlite_database(db_names: Tuple[str, ...],
                             david_home: str = None,
                             return_destination: bool = False) -> IO:
    """Downloads the main datasets available from google drive into the
    david_home directory.

    Parameters:
    ----------

    `db_names` (Tuple[str, ...], [1 or all: 3]):
        Choose a database file to download. It will download all the database
        files that are passed to db_name. Available keys ('v1', 'v2', 'unbox').

    `david_home` (str, default=DAVID_HOME):
        A user's home direcory. If it doesn't exists it will create the one
        and then download the requested files from the server (optional).

    `return_destination` (bool, default=False):
        Return the destination path of the file downloaded (optional)

    """
    if not david_home:
        # by default all files are saved in data_home
        david_home = get_data_home()
    gdrive = GoogleDriveDownloader(dirpath=david_home)
    sqlite_files = {
        k: v for k, v in GDRIVE_SQLITE_DATABASES.items() if k in db_names
    }
    for key, value in sqlite_files.items():
        db_files = [(g["name"], g["id"]) for g in value]
        for db_name, db_id in db_files:
            # here we just pass the the message to the instance for tqdm.
            gdrive.desc = f"Downloading {db_name} from server 📡"
            gdrive.filename = db_name
            gdrive.download(db_id)

    if return_destination:
        return gdrive._destination
