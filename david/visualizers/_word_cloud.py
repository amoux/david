import matplotlib.pyplot as plt
from wordcloud import WordCloud

from ..lang import SPACY_STOP_WORDS


def build_wordcloud(
        corpus: list,
        img_name: str = 'wordcloud',
        width: int = 1600,
        height: int = 600,
        max_words: int = 200,
        stop_words: list = None,
):
    """Build a word cloud image from text sequences."""

    if not stop_words:
        stop_words = SPACY_STOP_WORDS

    wordcloud = WordCloud(
        width=width,
        height=height,
        margin=3,
        max_words=max_words,
        max_font_size=150,
        random_state=62,
        background_color='black',
        stopwords=stop_words,
    ).generate(str(corpus))
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig.savefig(img_name, dpi=900)
