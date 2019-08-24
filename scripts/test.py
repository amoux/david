import re
import ast
import glob
import os
from os import path
from pathlib import Path

import astor
import spacy
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

# MULTITHREATING MODULES
# from multiprocessing import cpu_count
# from general_utils import apply_parallel, flattenlist

from math import ceil
from more_itertools import chunked
from typing import List, Callable, Union, Any
from pathos.multiprocessing import Pool, cpu_count


CORES = cpu_count()
nlp = spacy.load('en_core_web_sm')


def apply_parallel(func: Callable,
                   data: List[Any],
                   cpu_cores: int = None) -> List[Any]:
    '''Apply function to list of elements.
    Automatically determines the chunk size.
    '''
    if not cpu_cores:
        cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data)/cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size),
                                    chunksize=1)
    finally:
        pool.close()
        pool.join()
    return transformed_data


def flattenlist(list_of_lists: List[List[Any]]):
    return list(chain.from_iterable(list_of_lists))


def tokenize_docstring(texts):
    '''Apply tokenization using spacy to docstrings.
    '''
    tokens = []
    for token in nlp.tokenizer(texts):
        if not token.is_space:
            token_list.append(token.text.lower())
    yield tokens


def tokenize_code(texts):
    '''A very basic procedure for tokenizing code strings.
    '''
    for text in texts:
        yield RegexpTokenizer(r'\w+').tokenize(text)


def extract_docstring_pairs(blob):
    '''Extract (function/method, docstring) pairs from a given code blob.
    '''
    pairs = []
    try:
        module = ast.parse(blob)
        classes = [node for node in module.body if isinstance(
            node, ast.ClassDef)]
        functions = [node for node in module.body if isinstance(
            node, ast.FunctionDef)]
        for _class in classes:
            functions.extend(
                [node for node in _class.body
                    if isinstance(node, ast.FunctionDef)])

        for f in functions:
            source = astor.to_source(f)
            docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
            function = source.replace(ast.get_docstring(
                f, clean=False), '') if docstring else source

            pairs.append((f.name,
                          f.lineno,
                          source,
                          ' '.join(tokenize_code(function)),
                          ' '.join(tokenize_docstring(
                              docstring.split('\n\n')[0]))
                          ))
    except (AssertionError, MemoryError, SyntaxError, UnicodeEncodeError):
        pass
    yield pairs


def apply_to_bloblist(blob_list):
    '''Apply the function `extract_docstring_pairs` on a list of blobs.
    '''
    return [extract_docstring_pairs(blob) for blob in blob_list]


def iter_pairs(df, col_name, cores=CORES):
    yield flattenlist(apply_parallel(
        apply_to_bloblist, df[col_name].tolist(), cpu_cores=cores))


def get_docstring_pairs(df, col_name='content'):
    # creates a pairs column
    df['pairs'] = iter_pairs(df, col_name)


def flatten_pairs(df):
    df = df.set_index(['nwo', 'path'])['pairs'].apply(pd.Series).stack()
    df = df.reset_index()
    df.columns = ['nwo', 'path', '_', 'pair']


def extract_dataframe_metadata(df, col_name='pair'):
    '''Extract meta-data from dataframe
    '''
    df['function_name'] = df[col_name].apply(lambda p: p[0])
    df['lineno'] = df[col_name].apply(lambda p: p[1])
    df['original_function'] = df[col_name].apply(lambda p: p[2])
    df['function_tokens'] = df[col_name].apply(lambda p: p[3])
    df['docstring_tokens'] = df[col_name].apply(lambda p: p[4])


def format_dataframe(df):
    GIT_URL = 'https://github.com/{}/blob/master/{}#L{}'
    DF_COLS = ['nwo', 'path', 'function_name', 'lineno',
               'original_function', 'function_tokens', 'docstring_tokens']
    URL_COLS = ['nwo', 'path', 'lineno']

    df = df[DF_COLS]
    df['url'] = df[URL_COLS].apply(lambda col: GIT_URL.format(
        col[0], col[1], col[2]), axis=1)


def remove_duplicates(df):
    '''Remove observations where the same function appears more than once.
    '''
    before = len(df)
    df = df.drop_duplicates(['original_function', 'function_tokens'])
    after = len(df)
    print(f'removed: {(before-after):,} duplicate rows.')


def list_len(x):
    if not isinstance(x, list):
        return 0
    return len(x)


def seperate_docstring_tokens(df, col_name='docstring_tokens'):
    '''Separate functions w/o docstrings.
    Docstrings should be at least 3 words in the docstring
    to be considered a valid docstring.
    Returns: (with_docs, without_docs)
    '''
    with_docs = df[df[col_name].str.split().apply(list_len) >= 3]
    without_docs = df[df[col_name].str.split().apply(list_len) < 3]
    return (with_docs, without_docs)


def write_to(df, filename, path='data/processed_data/'):
    '''Helper function to write processed files to disk.
    '''
    if not path.exists(path):
        os.makedirs(path)
    out = Path(path)
    out.mkdir(exist_ok=True)

    no_docs = path.join(out, f'{filename}.docstring')
    tokens = path.join(out, f'{filename}.function')
    original = path.join(out, f'{filename}_original_function.json.gz')
    urls = path.join(out, f'{filename}.lineage')

    df.function_tokens.to_csv(tokens, index=False)
    df.original_function.to_json(original, orient='values', compression='gzip')

    if filename != 'without_docstrings':
        df.docstring_tokens.to_csv(no_docs, index=False)
    df.url.to_csv(urls, index=False)


def chunk_preprocess_csv(filename, sep=','):
    print('*step_1: reading parquet file.')
    df_chunks = pd.read_csv(filename, sep=sep, chunksize=10000)
    chunk_list = []
    for df in df_chunks:
        print('*step_2: getting docstring pairs.')
        get_docstring_pairs(df, col_name='content')
        print('*step_3: flattening docstring pairs.')
        flatten_pairs(df)
        print('*step_4: extracting dataframe metadata.')
        extract_dataframe_metadata(df, col_name='pair')
        format_dataframe(df)
        print('*step_5: removing duplicates.')
        remove_duplicates(df)
        print('DONE!\n\n')
        # append all dataframe columns
        chunk_list.append(df)
    # returns a preprocessed dataframe
    return pd.concat(chunk_list)


if __name__ == "__main__":

    df = chunk_preprocess_csv(filename='code_search_data.csv')
    with_docstrings, without_docstrings = seperate_docstring_tokens(df)

    grouped = with_docstrings.groupby('nwo')
    train, test = train_test_split(list(grouped), train_size=0.87,
                                   shuffle=True, random_state=8081)

    train, valid = train_test_split(train, train_size=0.82,
                                    random_state=8081)

    train = pd.concat([d for _, d in train]).reset_index(drop=True)
    valid = pd.concat([d for _, d in valid]).reset_index(drop=True)
    test = pd.concat([d for _, d in test]).reset_index(drop=True)

    print(f'train set num rows {train.shape[0]:,}')
    print(f'valid set num rows {valid.shape[0]:,}')
    print(f'test set num rows {test.shape[0]:,}')
    print(f'without docstring rows {without_docstrings.shape[0]:,}')

    # write to output files
    write_to(df=train, filename='train')
    write_to(df=valid, filename='valid')
    write_to(df=test, filename='test')
    write_to(df=without_docstrings, filename='without_docstrings')
