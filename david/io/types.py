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
