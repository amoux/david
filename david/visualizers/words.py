import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud


def build_wordcloud(corpus, img_name='wordcloud',
                    width=1600, height=600,
                    max_words=200, stop_words=STOPWORDS
                    ):
    '''Build a word cloud image from text sequences.
    '''
    if stop_words:
        STOPWORDS = stop_words

    wordcloud = WordCloud(
        width=width,
        height=height,
        margin=3,
        max_words=max_words,
        max_font_size=150,
        random_state=62,
        background_color='black',
        stopwords=STOPWORDS).generate(str(corpus))

    img_name = img_name + '.png'
    fig = plt.figure(1)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    fig.savefig(img_name, dpi=900)
