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
    """Computes cosine similarity, as the normalized dot product of X and Y."""
    vectorizer = vectorizer.toarray()[0]
    features = features.toarray()  # 2 dimensional ndarray.
    similar = numpy.dot(vectorizer, features.T)
    sim_matrix = similar.argsort()[::-1][:num_results]
    doc_scores = [(i, round(similar[i], round_float)) for i in sim_matrix]
    return doc_scores


class SimilarDocumentMatrix:

    def __init__(
        self,
        raw_doc: List[str] = None,
        num_results: int = 3,
        ngram: Tuple[int, int] = (1, 1),
    ):
        """Get the most similar documents from top occurrences of terms.

        Parameters:
        ----------

        `raw_doc` (List[str], default=None):
            The corpus is expected to be a sequence of strings or bytes items
            are expected to be analysed directly.

        `ngram` (Tuple[int, int], default=(1, 1)):
            Choose an ngram range for the document frequency.

        TODO: Add examples to the documentation.

        """
        self.raw_doc = raw_doc
        self.num_results = num_results
        self.ngram = ngram
        self.queries = []
        self.vectorizer = None
        self.features = None

    def clear_queries(self, query_or_queries=None):
        """Clear the queries accumulated in a session."""
        if self.queries is not None:
            self.queries.clear()
        if query_or_queries is not None:
            if isinstance(query_or_queries, list):
                self.add_queries(query_or_queries)
            elif isinstance(query_or_queries, str):
                self.add_query(query_or_queries)

    def add_query(self, query: str):
        """Extend the existing query list from a single string."""
        if isinstance(query, str):
            self.queries.append(query)

    def add_queries(self, queries: List[str]):
        """Extend the existing query list from an iterable of strings."""
        if isinstance(queries, list):
            self.queries.extend(queries)

    def fit_features(self, ngram: Tuple[int, int] = None,
                     feature: str = "tfidf",
                     min_freq: float = 0.0,
                     max_freq: float = 1.0) -> Dict[str, str]:
        """Fit the vectorizer and feature matrix on the raw documents"""
        if ngram is None:
            ngram = self.ngram
        self.vectorizer, self.features = build_feature_matrix(
            self.raw_doc, feature=feature, ngram=ngram,
            min_freq=min_freq, max_freq=max_freq)

    def iter_similar(self, num_results: int = None, queries: List[str] = None):
        """Yields an iterable of key value pairs."""
        if queries is None:
            queries = self.queries
        if num_results is None:
            num_results = self.num_results

        doc_matrix = self.vectorizer.transform(queries)
        for i, q in enumerate(queries):
            sparse_vec = doc_matrix[i]
            doc_scores = cosine_similarity(sparse_vec, self.features,
                                           num_results)
            for doc_id, score in doc_scores:
                text = raw_doc[doc_id]
                yield {"q_id": i, "doc_id": doc_id, "sim": score, "text": text}
