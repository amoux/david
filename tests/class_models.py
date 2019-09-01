import re

import pandas as pd


class DavidDataFrameBase(pd.DataFrame):
    ABOUT_BASE = 'I Load data structure types to load your files.'

    def __init__(self, data_structure):
        super().__init__(data_structure)


class JsonDataFrame(DavidDataFrameBase):
    JSON_PATH = None

    def __init__(self, corpus_path):
        super().__init__(pd.read_json(
            corpus_path, encoding='utf-8', lines=True))
        self.JSON_PATH = corpus_path

    def get_json_path(self):
        return self.JSON_PATH


class TextMetrics(JsonDataFrame):
    SENTI_LABELS = ('positive', 'negative', 'neutral')

    def __init__(self, corpus_path: str):
        super().__init__(corpus_path)

    def get_labels(self):
        return self.SENTI_LABELS


class TextPreprocess(JsonDataFrame):

    def __init__(self, corpus_path: str):
        super().__init__(corpus_path)

    def corpus_filepath(self):
        return self.CORPUS_PATH

    def obj_to_dict(self, df_obj: object):
        return df_obj.to_dict(orient='index')

    def missing_values(self):
        return self.isnull().sum()

    def remove_whitespaces(self, text: str):
        return re.sub(r'(\s)\1{1,}', ' ', text).strip()
