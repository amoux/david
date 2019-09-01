
from ._metric import TextMetrics
from ._prep import TextPreprocess


class TextPipeline(TextMetrics, TextPreprocess):
    '''TextPipeline class with metrics and preprocessing methods.
    '''
    def __init__(self, corpus_path):
        super().__init__(corpus_path)
