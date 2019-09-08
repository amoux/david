
import logging
import os

import numpy as np
import pandas as pd
import requests
import spacy
import tensorflow as tf
import tensorflow_hub as tfHub
from IPython.display import HTML
from sklearn import preprocessing
from spacy import displacy
from spacy.lang.en import English

logging.getLogger('tensorflow').disabled = True


nlp = spacy.load('en_core_web_lg')


class GoogleDriveDownloader:
    URL = 'https://docs.google.com/uc?export=download'
    DIR = 'data'

    def __init__(self, file_id: str, file_name='temp.p', chunk_size=32_768):
        self.file_id = file_id
        self.file_name = file_name
        self.destination = os.path.join(self.DIR, file_name)
        self.chunk_size = chunk_size

    def _confirm_token(self, response):
        for (key, val) in response.cookies.items():
            if key.startswith('download_warning'):
                return val
        return None

    def _save_content(self, response):
        if not os.path.exists(self.DIR):
            os.makedirs(self.DIR)
        with open(self.destination, 'wb') as f:
            for chunk in response.iter_content(self.chunk_size):
                if chunk:
                    f.write(chunk)

    def _load_dataframe(self):
        df = pd.read_pickle(self.destination)
        return df

    def download_dataframe(self, stream=True, load_df=False):
        session = requests.Session()
        response = session.get(
            self.URL, params={'id': self.file_id}, stream=stream)
        token = self._confirm_token(response)
        if token:
            params = {'id': self.file_id, 'confirm': token}
            response = session.get(self.URL, params=params, stream=stream)
        self._save_content(response)
        if load_df:
            return self._load_dataframe()


# def create_batch(self, sentence_list):
#     """Create a batch for a list of sentences
#     """
#     embeddings_batch = []

#     for sent in sentence_list:
#         embeddings = []
#         sent_tokens = sent_tokenize(sent)
#         word_tokens = [word_tokenize(w) for w in sent_tokens]
#         tokens = [value for sublist in word_tokens for value in sublist]
#         tokens = [token for token in tokens if token != '']

#         for token in tokens:
#             embeddings.append(self.embdict.tok2emb.get(token))

#         if len(tokens) < self.max_sequence_length:
#             pads = [np.zeros(self.embedding_dim) for _ in range(
#                 self.max_sequence_length - len(tokens)
#             )]
#             embeddings = pads + embeddings
#         else:
#             embeddings = embeddings[-self.max_sequence_length:]

#         embeddings = np.asarray(embeddings)
#         embeddings_batch.append(embeddings)

#     embeddings_batch = np.asarray(embeddings_batch)
#     return embeddings_batch


class SchoolMember:
    '''Represents any school member.'''

    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('(Initialized SchoolMember: {})'.format(self.name))

    def tell(self):
        '''Tell my details.'''
        print('Name:{} Age:{}'.format(self.name, self.age), end=' ')


class Teacher(SchoolMember):
    '''Represents a teacher.'''

    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: {:d}'.format(self.salary))


class Student(SchoolMember):
    '''Represents a student.'''

    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Marks: {:d}'.format(self.marks))


t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)
# prints a blank line
print()
members = [t, s]
for member in members:
    # Works for both Teachers and Students
    member.tell()
