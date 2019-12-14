from typing import Any, Dict, List, Tuple, TypeVar

import numpy
import sklearn
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import (CountVectorizer, TfidfVectorizer,
                                             VectorizerMixin)

Vector = List[float]
SparseRowMatrix = TypeVar("SparseMatrix", csr_matrix, Tuple[Vector, ...])


def docsize_mb(doc: List[str]) -> float:
    """Get the size in megabytes(MB) of a List[str] type iterable."""
    return sum(len(seq.encode("utf-8")) for seq in doc) / 1e6


def build_feature_matrix(
        raw_doc: List[str],
        feature: str = "tfidf",
        ngram: Tuple[int, int] = (1, 1),
        min_freq: float = 0.0,
        max_freq: float = 1.0) -> Tuple[VectorizerMixin, SparseRowMatrix]:
    """Convert a collection of text documents to a sparse matrix.

    TODO: Improve, finish add examples to the documentation.

    Parameters:
    ----------

    `raw_doc` (List[str]):
        The corpus is expected to be a sequence of strings or bytes items
        are expected to be analysed directly.

    `feature` (str):
        Select a feature type 'count' or 'tfidf' for the raw document.
            - `count`: Convert a collection of text documents to a matrix
                of token counts.
            - `tfidf`: Transform a count matrix to a normalized tf or tf-idf
                representation.

    `ngram` (Tuple[int, int], default=(1, 1)):
        Choose an ngram range for the document frequency.

    `min_freq` (float or int, default=0.0):
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. Parameter is
        linked to max_df from scikit-learn CountVectorizer() method.

    Returns:
        Tuple[VectorizerMixin, SparseRowMatrix]

    """
    feature = feature.lower().strip()
    if feature == "count":
        vectorizer = CountVectorizer(binary=False, min_df=min_freq,
                                     max_df=max_freq, ngram_range=ngram)
    if feature == "tfidf":
        vectorizer = TfidfVectorizer(min_df=min_freq, max_df=max_freq,
                                     ngram_range=ngram)
    else:
        raise ValueError(f"You entered {feature}, please \
            choose one: 'count' or 'tfidf'")
    feature_matrix = vectorizer.fit_transform(raw_doc).astype(float)
    return (vectorizer, feature_matrix)


def cosine_similarity(vectorizer: VectorizerMixin,
                      features: SparseRowMatrix,
                      num_results: int = 3,
                      round_float: int = 4) -> List[Tuple[int, float]]:
    """Computes cosine similarity, as the normalized dot product of X and Y.

    TODO: Improve, finish add examples to the documentation.
    """
    vectorizer = vectorizer.toarray()[0]
    features = features.toarray()  # 2 dimensional ndarray.
    similar = numpy.dot(vectorizer, features.T)
    sim_matrix = similar.argsort()[::-1][:num_results]
    doc_scores = [(i, round(similar[i], round_float)) for i in sim_matrix]
    return doc_scores


def get_similar_docs(queries: Dict[str, List[str]],
                     raw_doc: List[str],
                     num_results: int,
                     vectorizer: VectorizerMixin,
                     features: SparseRowMatrix) -> List[Dict[Any, str]]:
    """Get the most similar documents from top occurrences of terms.

    Parameters:
    ----------

    `queries`: (Dict[str, List[str]]):
        An iterable of sequences to query for similarities from the raw
        documents used with `david.cosine.build_feature_matrix` method.

    TODO: Improve, finish add examples to the documentation.

    """
    doc_matrix = vectorizer.transform(queries)
    similarities = list()
    for i, query in enumerate(queries):
        sparse_vec = doc_matrix[i]
        doc_scores = cosine_similarity(sparse_vec, features, num_results)
        for doc_id, score in doc_scores:
            text = raw_doc[doc_id]
            similarities.append({'doc_id': doc_id, 'sim': score, 'text': text})
    return similarities


def compute_similardocs(queries: Dict[str, List[str]],
                        raw_doc: List[str],
                        num_results: int,
                        ngram: Tuple[int, int],
                        min_freq: int,
                        max_freq: int) -> List[Dict[float, str]]:
    """Test all the methods above in one method. I was thinking of getting
    rid of all methods and put them all in one call but then I found that
    I could make multiple's like this one. And adapt them to whatever I feel
    like. Maybe I need to add some preprocessing or web components between
    the inputs and the outputs. This is an example method for how both cosine
    methods in this module can be used to fit a specific environment.
    """
    vectorizer, features = build_feature_matrix(raw_doc, "tfidf", ngram,
                                                min_freq, max_freq)
    doc_matrix = vectorizer.transform(queries)
    similarities = list()
    for i, q in enumerate(queries):
        sparse_vec = doc_matrix[i]
        doc_scores = cosine_similarity(sparse_vec, features, num_results)
        for doc_id, score in doc_scores:
            text = raw_doc[doc_id]
            similarities.append({"sim": score, "text": text})
    return similarities
