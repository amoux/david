from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy
import sklearn
from david.text.prep import preprocess_doc
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
                      top_k: int = 3,
                      round_float: int = 4) -> List[Tuple[int, float]]:
    """Computes cosine similarity, as the normalized dot product of X and Y."""
    vectorizer = vectorizer.toarray()[0]
    features = features.toarray()  # 2 dimensional ndarray.
    similar = numpy.dot(vectorizer, features.T)
    sim_matrix = similar.argsort()[::-1][:top_k]
    doc_scores = [(i, round(similar[i], round_float)) for i in sim_matrix]
    return doc_scores


class SimilarDocuments:

    def __init__(
        self,
        raw_doc: List[str] = None,
        top_k: int = 3,
        ngram: Tuple[int, int] = (1, 1),
        feature: str = "tfidf",
    ):
        """Get the most similar documents from top occurrences of terms.

        Parameters:
        ----------

        `raw_doc` (List[str], default=None):
            The corpus is expected to be a sequence of strings or bytes items
            are expected to be analysed directly.

        `top_k` (Optional[int], default=None):
            Get the top k (number) of similar documents.

        `ngram` (Tuple[int, int], default=(1, 1)):
            Choose an ngram range for the document frequency.

        """
        self.raw_doc = raw_doc
        self.top_k = top_k
        self.ngram = ngram
        self.feature = feature
        self.queries = []
        self.vectorizer = None
        self.features = None

    def clear_queries(self) -> None:
        self.queries.clear()

    def add_query(self,
                  query: Union[str, List[str]] = None,
                  clear_first: bool = False) -> None:
        """Extends or replaces the existing queries in an instance.

        Parameters:
        ----------

        `query` (Union[str, List[str]]):
            Add queries from a single string or and iterable of sequences.

        `clear_first` (bool, default=False):
            If True, it clears the existing items in the instance attribute
            `SimilarDocuments.queries`.
        """
        if clear_first:
            self.clear_queries()
        if query:
            if isinstance(query, str):
                self.queries.append(query)
            elif isinstance(query, list):
                self.queries.extend(query)

    def learn_vocab(self, min_freq: float = 0.0, max_freq: float = 1.0):
        """Learn vocabulary equivalent to fit followed by transform."""
        self.vectorizer, self.features = build_feature_matrix(
            raw_doc=self.raw_doc, feature=self.feature, ngram=self.ngram,
            min_freq=min_freq, max_freq=max_freq)

    def iter_similar(self,
                     top_k: Optional[int] = None,
                     query: Optional[Union[str, List[str]]] = None,
                     clear_first: bool = False) -> Dict[str, str]:
        """Iterate over all the queries returning the most similar document.

        Parameters:
        ----------

        `top_k` (Optional[int], default=None):
            Get the top k (number) of similar documents.

        `query` (Optional[Union[str, List[str]]]):
            Add queries from a single string or and iterable of sequences.

        Returns (Dict[str, str]): Yields a dictionary of key value pairs.

        """
        top_k = self.top_k if not top_k else top_k
        if query:
            self.add_query(query=query, clear_first=clear_first)

        doc_matrix = self.vectorizer.transform(self.queries)
        for i, q in enumerate(self.queries):
            sparse_vec = doc_matrix[i]
            doc_scores = cosine_similarity(sparse_vec, self.features, top_k)
            for doc_id, score in doc_scores:
                text = self.raw_doc[doc_id]
                yield {"q_id": i, "doc_id": doc_id, "sim": score, "text": text}
