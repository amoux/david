import heapq

import spacy

from ..lang.stopwords import SPACY_STOP_WORDS


def spacy_summarizer(raw_doc):
    """Summarizes a document of sents using Spacy."""

    if isinstance(raw_doc, list):
        raw_doc = str(raw_doc)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_doc)
    word_freq = {}
    for word in doc:
        if word.text not in SPACY_STOP_WORDS:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_frequency = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = (word_freq[word] / max_frequency)

    sents = [sent for sent in doc.sents]
    sent_scores = {}
    for sent in sents:
        for word in sent:
            if word.text.lower() in word_freq.keys():
                if len(sent.text.split(" ")) < 30:
                    if sent not in sent_scores.keys():
                        sent_scores[sent] = word_freq[word.text.lower()]
                    else:
                        sent_scores[sent] += word_freq[word.text.lower()]

    highest_rating = heapq.nlargest(7, sent_scores, key=sent_scores.get)
    summarized = " ".join([sent.text for sent in highest_rating])
    return summarized
