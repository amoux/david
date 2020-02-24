import pickle
from pathlib import Path
from typing import IO, Any


class DataIO:
    @staticmethod
    def save_data(file_name: str, data_obj: Any, dirname="data") -> IO:
        path = Path(dirname).absolute()
        if not path.exists():
            path.mkdir(parents=True)
        file = path.joinpath(file_name)
        with file.open("wb") as pkl:
            pickle.dump(data_obj, pkl, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_data(file_name: str, dirname="data") -> Any:
        path = Path(dirname).absolute()
        file = path.joinpath(file_name)
        with file.open("rb") as pkl:
            return pickle.load(pkl)
