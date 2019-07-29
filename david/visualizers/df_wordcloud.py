# NOTE TO SELF: HERE I AM PLANNING TO WRAP ALL THE VISUALIZING FUNCTIONS
# THESE TOOLS HAVE A LOT OF CODE THAT SHOULD BE HIDDEN IN THE BACK
# SO WITH THAT BEING SAID PUT ALL THE VISUALIZER FUNCTIONS HERE!

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS


def make_wordcloud(corpus, filename):
    wordcloud = WordCloud(
        width=1600,
        height=600,
        margin=3,
        max_words=200,
        max_font_size=150,
        random_state=62,
        background_color='black',
        stopwords=STOPWORDS).generate(str(corpus))

    img_path = filename + '.png'
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig.savefig(img_path, dpi=900)
