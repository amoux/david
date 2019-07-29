import os
import dataset
import sqlite3
import pandas as pd
from tqdm import tqdm


def arrange_index(df, rm_index: int, col_name: str, to_index: int):
    columns = list(df.columns.values)
    columns.pop(rm_index)
    columns.insert(to_index, col_name)
    df = df[columns]
    return df


def preprocess_dataframe(filepath: str, filename: str):
    """
    * Creates a video_id column while adding the videoids
    to the row from the filename parameter.
    * Splits the 'cid' column, creates a new column 'cid_reply',
    and arranges the index column in relation to 'cid'.
    """
    df = pd.read_json(filepath, encoding='utf-8', lines=True)
    df['video_id'] = filename.strip('.json')
    df[['cid', 'cid_reply']] = pd.DataFrame(
        [x.split('.') for x in df['cid'].tolist()]
    )
    df = arrange_index(df, 5, 'cid_reply', 2)
    return df


def init_dataframe_batches(filepaths, videoids):
    """Iterates over the files to dataframes
    """
    for filepath, vidid in zip(filepaths, videoids):
        yield preprocess_dataframe(filepath, vidid)


def json_tosql(df, table_name: str, db_name: str):
    """Inserts Data to SQLite from a Dataframe

    PARAMETERS
    ----------
    df : (object)
        A pandas.Dataframe containing the rows
        specified in the table dictionary.

    table_name : (str)
        The name of the table, if it exists it will append rows
        containing new data. Otherwise it creates a new table name
        and assigns Datatypes according to the df.

    db_name : (str)
        The connection name of the datbase
    """
    sqlite_address = f"sqlite:///{db_name}.db"
    db = dataset.connect(sqlite_address)
    db.begin()
    sql_table = db[table_name]
    try:
        for _, col in df.iterrows():
            sql_table.insert(
                dict(
                    author=col['author'],
                    cid=col['cid'],
                    cid_reply=col['cid_reply'],
                    text=col['text'],
                    time=col['time'],
                    video_id=col['video_id']
                )
            )
        db.commit()
    except RuntimeError:
        db.rollback()


def get_directory_files(dirname):
    """
    * Gets all the paths to all files found in the root
    directory.
    * Returns two lists containing the paths & file names
    """
    filepaths = []
    filenames = []
    for (dirpath, _, filenames) in os.walk(dirname):
        for filename in filenames:
            if filename.endswith('.json'):
                filepaths.append(dirpath + '/' + filename)
                filenames.append(filename)
    return filepaths, filenames


# testing purposes only: creates a new database file
conn = sqlite3.connect("ycc_web_sqlite2.db")
conn.close()


if __name__ == '__main__':

    json_files, videoids = get_directory_files('downloads')
    batches = init_dataframe_batches(json_files, videoids)

    for batch in tqdm(batches):
        json_tosql(batch, 'videos', 'ycc_web_sqlite2')
