"""BOW for binary classification:

- Simple Bag-of-words model mainly to show the functionality
    of the `david.text.tokenizer.WordTokenizer` class.

Expected Results:
    1. Tweet: [blog] using nullmailer and mandrill for your ubuntu
              linux server outboud mail http://bit.ly/zjhok7 #plone
    Prediction(label=app, weight=-0.010066255795677341)

    2. Tweet: ¿En donde esta su remontada Mandrill?
    Prediction(label=other, weight=-0.019214811313018675)
"""

from typing import Dict, List, Tuple
from nptyping import Array

import numpy as np
from david.text import normalize_whitespaces, unicode_to_ascii
from david.text.tokenizers import WordTokenizer


class Dataset(WordTokenizer):
    def __init__(
        self,
        data_path: str,
        preserve_case: bool = False,
        reduce_len: bool = False,
        strip_handles: bool = False,
    ):
        super().__init__(preserve_case, reduce_len, strip_handles)
        self.sentences: List[str] = self._load_dataset(data_path)
        self.bow = Dict[str, float] = {}
        self.num_words: int = 0
        # vocabulary instance attributes:
        self.word2count: Dict[str, int] = {}
        self.word2index: Dict[str, int] = {}
        self.index2word: Dict[int, str] = {}

    def _load_dataset(self, dataset_path: str) -> List[str]:
        sents = []
        with open(dataset_path, "r") as text_file:
            for line in text_file:
                text = unicode_to_ascii(line.strip())
                text = normalize_whitespaces(text)
                sents.append(text)
        return sents

    def _build_bow(self) -> None:
        # Builds a bag-of-words dict from unique tokens in the vocabulary.
        size = np.array([i + 1 for i in self.word2count.values()]).sum()
        for word, count in self.word2count.items():
            self.bow[word] = self.ln_step_func(count, k=size)

    def ln_step_func(self, n: int, k: int) -> float:
        """LN step-function.

        Computes the logarithm of N (Eulers number).
        """
        weight = np.log((n + 1) / k)
        return weight

    def build_vocab(self) -> None:
        """Builds the vocabulary from unique words across all sentences."""
        for sent in self.sentences:
            toks = self.tokenize(sent)
            for token in toks:
                self.add_word(token.lower())
        self._build_bow()

    def __repr__(self):
        return f"< Dataset(samples={len(self.sentences)})>"


def encode(tokens: List[str], bow: Dict[str, float]) -> Array:
    embedding = np.array([bow[token] for token in tokens if token in bow])
    return embedding


def predict(sequence: str, x: Dataset, y: Dataset) -> Tuple[float, str]:
    seq_tokens = WordTokenizer().tokenize(sequence.lower())
    w1 = 1 / encode(seq_tokens, x.bow).sum()
    w2 = 1 / encode(seq_tokens, y.bow).sum()

    score, label = (0, "{}")
    if w1 > w2:
        score, label = (w1, label.format("app"))
    else:
        score, label = (w2, label.format("other"))
    return score, label


def main():
    # Load both datasets of tweets (app vs not-app) for this demo.
    X_data = Dataset("app_tweets.txt")
    X_data.build_vocab()
    y_data = Dataset("other_tweets.txt")
    y_data.build_vocab()

    # Display one sentence embedding:
    xsent0_string = X_data.sentences[0]
    xsent0_embedd = encode(X_data.tokenize(xsent0_string), bow=X_data.bow)
    print("embedding sentence_id(0): {}".format(xsent0_embedd))

    # Test model with x and y tweets:
    tweet_app = (
        "[blog] using nullmailer and mandrill for your ubuntu "
        "linux server outboud mail http://bit.ly/zjhok7 #plone"
    )
    tweet_other = "¿En donde esta su remontada Mandrill?"

    # Display the predictions from both tweet examples.
    for sequence in [tweet_app, tweet_other]:
        score, label = predict(sequence, X_data, y_data)
        print(
            "Tweet: {}\nPrediction(label={}, weight={})\n".format(
                sequence, label, score
            )
        )


if __name__ == "__main__":
    main()
