import matplotlib.pyplot as plt
from wordcloud import WordCloud

from ..lang import SPACY_STOP_WORDS


def build_wordcloud(
    doc: list,
    img_name: str = "wordcloud",
    width: int = 1600,
    height: int = 600,
    margin=3,
    max_words: int = 200,
    max_font_size=150,
    image_dpi=900,
    random_state=62,
    background_color="black",
    stop_words: list = None,
):
    """Build a word cloud image from text sequences."""
    if not stop_words:
        stop_words = SPACY_STOP_WORDS
    wordcloud = WordCloud(
        width=width,
        height=height,
        margin=margin,
        max_words=max_words,
        max_font_size=max_font_size,
        random_state=random_state,
        background_color=background_color,
        stopwords=stop_words,
    ).generate(str(doc))
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    fig.savefig(img_name, dpi=image_dpi)
