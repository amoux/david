from typing import List, Optional, Set, Union

import spacy

from ..lang.stopwords import SPACY_STOP_WORDS


def summarizer(
    document: Union[List[str], str],
    nlargest: int = 7,
    min_sent_toks: int = 30,
    spacy_model: str = "en_core_web_sm",
    stop_words: Optional[Union[List[str], Set[str]]] = None,
) -> str:
    """Text summarizer built on top of spaCy.

    `document` (Union[List[str], str]):
        An iterable list of sequences or text of sequences.

    `stop_words` (Optional[Union[List[str], Set[str]]], default=None):
        A list or set of string stop-words to be used for word frequency.
        Word frequency is measured for each word in the document not in
        the stop-words. If parameter is left as None, then SPACY_STOP_WORDS
        will be used.

    Notes: For improving results from the summarizer, e.g,. summarizing
        texts extracted from the web - it's recommended to encode unicode
        strings to ascii and normalizing whitespaces. Methods available in
        david's API that achieve this: `david.text.prep.unicode_to_ascii`
        and `david.text.prep.normalize_whitespace`.

    """
    nlp = spacy.load(spacy_model)
    spacydoc = None

    if isinstance(document, list):
        spacydoc = nlp(" ".join(document))
    elif isinstance(document, str):
        spacydoc = nlp(document)

    if stop_words is None:
        stop_words = SPACY_STOP_WORDS

    frequency = dict()  # Word level frequency.
    for token in spacydoc:
        token = token.text
        if token not in stop_words:
            if token not in frequency:
                frequency[token] = 1
            else:
                frequency[token] += 1

    # Normalize word frequency distribution (val:int => float)
    for word in frequency:
        frequency[word] = frequency[word] / max(frequency.values())

    scores = dict()  # Sentence level scores.
    for sentence in [sent for sent in spacydoc.sents]:
        for word in sentence:
            word = word.text.lower()
            if word in frequency:
                if len(sentence.text.split()) < min_sent_toks:
                    if sentence not in scores:
                        scores[sentence] = frequency[word]
                    else:
                        scores[sentence] += frequency[word]

    # Find the n[:n] largest sentences from the scores.
    sentences = sorted(scores, key=scores.get, reverse=True)[:nlargest]
    # Return the sentences as sequences of strings.
    summary = " ".join([i.text for i in sentences])
    return summary
