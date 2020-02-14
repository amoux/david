"""
Types for data structures used in david (experimental).

-------------------------------------------------------
Type hints declared in this file are Experimental Before adding
a bunch of made-up types I need to make sure I follow the correct
patterns and use other NLP libraries who currently have adapted
many of the namespaces from the typing module.
"""
from typing import IO, Dict, Generator, Generic, List, NewType, Text, TypeVar

import pandas
import torch

# examples of aliases types:
Token = List[str]  # token types (I have wanted a Token type!)
TokenIterable = List[Token]  # this is a List of tokens.
Vector = List[float]
DataArray = TypeVar("DataArray", torch.Tensor, Dict[str, torch.Tensor])
TensorType = Dict[str, DataArray]
# new defined types
YoutubeId = NewType("YoutubeId", str)
DataFrameInstance = NewType("DataFrameInstance", pandas.DataFrame)
SeriesInstance = NewType("SeriesInstance", pandas.Series)
