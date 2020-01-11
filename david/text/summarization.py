
from typing import List, Optional, Set, Union
from heapq import nlargest
import spacy

from ..lang.stopwords import SPACY_STOP_WORDS


def spacy_summarizer(document: Union[List[str], str],
                     stop_words: Optional[Union[List[str], Set[str]]] = None) -> str:
    """Summarizes a document of sents using Spacy."""
    if isinstance(document, list):
        document = str(document)
    if stop_words is None:
        stop_words = SPACY_STOP_WORDS

    nlp = spacy.load("en_core_web_sm")
    spacydoc = nlp(document)

    frequency = dict()
    for token in spacydoc:
        token = token.text
        if token not in stop_words:
            if token not in frequency:
                frequency[token] = 1
            else:
                frequency[token] += 1

    for word in frequency:
        # Normalize word[freq:int]:float = (word:int/max(word:int))
        frequency[word] = (frequency[word]/max(frequency.values()))

    scores = dict()
    for sentence in [sent for sent in spacydoc.sents]:
        for word in sentence:
            word = word.text.lower()
            if word in frequency:
                if len(sentence.text.split(" ")) < 30:
                    if sentence not in scores:
                        scores[sentence] = frequency[word]
                    else:
                        scores[sentence] += frequency[word]

    top_scores = nlargest(7, scores, key=scores.get)
    summarized = " ".join([i.text for i in top_scores])
    return summarized
