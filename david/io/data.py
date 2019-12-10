"""Types for NLP context experimental.

NOTE: The type hints declared in this file are Experimental
Before adding a bunch of made-up types I need to make sure
I follow the correct patterns and use other NLP libraries who
currently have adapted many of the namespaces from the typing
module. One good example is AllenNLP.

Example from AllenNLP `allennlp.data.vocabulary`

Examples:
---------

class Vocabulary:
    def __init__(
        self,
        counter: Dict[str, Dict[str, int]] = None,
        min_count: Dict[str, int] = None,
        max_vocab_size: Union[int, Dict[str, int]] = None,
        non_padded_namespaces: Iterable[str] = DEFAULT_NON_PADDED_NAMESPACES,
        pretrained_files: Optional[Dict[str, str]] = None,
        only_include_pretrained_words: bool = False,
        tokens_to_add: Dict[str, List[str]] = None,
        min_pretrained_embeddings: Dict[str, int] = None,
        padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
        oov_token: Optional[str] = DEFAULT_OOV_TOKEN,
    ) -> None:
        ...

* An example of an optional argument that takes a string:

    >>> name: Optional[str] = None # Same as name: str = None.

Below are some example that could be implemented:

"""
from typing import IO, Dict, Generator, Generic, List, NewType, Text, TypeVar

import pandas
import torch

# Examples of aliases types:
Token = List[str]  # token types (I have wanted a Token type!)
TokenIterable = List[Token]  # this is a List of tokens.
Vector = List[float]

DataArray = TypeVar("DataArray", torch.Tensor, Dict[str, torch.Tensor])
TensorType = Dict[str, DataArray]

# New defined types
YoutubeId = NewType("YoutubeId", str)
DataFrameInstance = NewType("DataFrameInstance", pandas.DataFrame)
SeriesInstance = NewType("SeriesInstance", pandas.Series)
