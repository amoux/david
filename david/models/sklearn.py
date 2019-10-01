# from gensim.matutils import hellinger
from typing import Any, AnyStr, Dict, Iterable, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_feature_matrix(docs,
                         feature_type='frequency',
                         ngram=(1, 1),
                         min_freq=0.0,
                         max_freq=1.0):
    '''Convert a collection of text documents
    to a matrix of token counts.

    Parameters:
    ----------

    `docs` : (str), sequences of strings
        The corpus is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    feature_type: (str)
        Select a feature type (binary|frequency|tfidf) for the document.

    `ngram` : tuple, int
        choose a ngram for the document frequency.

    `min_freq` : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. Parameter is
        linked to max_df from scikit-learn CountVectorizer() method

    `max_freq` : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
    '''
    feature_type = feature_type.lower().strip()
    if (feature_type == 'binary'):
        vectorizer = CountVectorizer(
            binary=True,
            min_df=min_freq,
            max_df=max_freq,
            ngram_range=ngram)
    if (feature_type == 'frequency'):
        vectorizer = CountVectorizer(
            binary=False,
            min_df=min_freq,
            max_df=max_freq,
            ngram_range=ngram)
    if (feature_type == 'tfidf'):
        vectorizer = TfidfVectorizer(
            min_df=min_freq,
            max_df=max_freq,
            ngram_range=ngram)
    else:
        raise Exception(
            'Wrong feature entered, choose: binary, frequency, tfidf')
    feature_matrix = vectorizer.fit_transform(docs).astype(float)
    return (vectorizer, feature_matrix)


def cosine_similarity(vec, features, num_results=3):
    vec = vec.toarray()[0]
    features = features.toarray()
    similar = np.dot(vec, features.T)
    top_similar = similar.argsort()[::-1][:num_results]
    top_scores = [(i, round(similar[i], 3)) for i in top_similar]
    return top_scores


def get_similar_docs(
        queries: List[str],
        docs: List[str],
        num_results: int,
        tfid_vectorizer: Any,
        tfid_features: Any) -> Iterable[Dict]:
    '''
    Cosine Similarity Analysis.

    Parameters:
    ----------
    `docs` : (type=List[str])
    `num_results` : (type=int)
    `tfid_vectorizer` :
        sklearn text TfidfVectorizer
    `tfid_features` :
        sklearn csr_matrix from TfidVectorizer
    '''
    transform_vec = tfid_vectorizer.transform(queries)

    matches = []
    for i, query in enumerate(queries):
        vec_features = transform_vec[i]
        similar_docs = cosine_similarity(vec=vec_features,
                                         features=tfid_features,
                                         num_results=num_results)
        for doc, score in similar_docs:
            text = docs[doc]
            matches.append({
                'doc_id': doc+1,
                'text': text,
                'similarity': score,
                'query': query,
            })
    return matches
