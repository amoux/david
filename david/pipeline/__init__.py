
from .base import DavidDataFrame
from ._metric import TextMetrics
from ._prep import TextPreprocess


class Pipeline(DavidDataFrame, TextMetrics, TextPreprocess):
    pass
