import random
import string
import unicodedata
import warnings

import spacy
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

# nltk packages required for this notebook:
# nltk.download('popular', quiet=True)
# nltk.download('punkt')
# nltk.download('wordnet')

warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_lg')

GREET_INTENTS = ('hello', 'hi', 'greeting',
                 'sup', 'what\'s is up', 'hey')

GREET_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello',
                   'I am glad you are talking to me!']


def encode_ascii(tokens: list):
    ascii_tokens = []
    for token in tokens:
        ascii_ = unicodedata.normalize('NFKD', token).encode(
            'ascii', 'ignore').decode('utf-8', 'ignore')
        ascii_tokens.append(ascii_)
    return ascii_tokens


def get_textdocs(fp: str, errors='ignore'):
    with open(fp, errors='ignore') as text_file:
        text_docs = []
        for text in text_file:
            if len(text) > 1:
                tokens = [s for s in text.strip().split()]
                string = ' '.join(encode_ascii(tokens))
                text_docs.append(string)
        return text_docs


def sentence_tokenizer(raw_texts: list):
    sent_tokens = []
    for text in raw_texts:
        doc = nlp(text)
        for token in doc.sents:
            sent_tokens.append(token.text)
    return sent_tokens


def token_lemmatizer(tokens: list):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(token) for token in tokens]


def string_tokenizer(text: str):
    punct = dict((ord(p), None) for p in string.punctuation)
    return token_lemmatizer(
        wordpunct_tokenize(text.translate(punct)))


def greeting_response(text: str, intents: set):
    greeting = None
    for token in text.split():
        if token.lower() in intents:
            greeting = random.choice(GREET_RESPONSES)
    return greeting


def similar_response(text: str, sents: list, stop_words=STOP_WORDS):
    sents.append(text)
    tfid = TfidfVectorizer(tokenizer=string_tokenizer,
                           stop_words=list(stop_words))
    docs = tfid.fit_transform(raw_documents=sents)
    similar_cos = cosine_similarity(X=docs[-1], Y=docs)
    similar_idx = similar_cos.argsort()[0][-2]
    flatten_cos = similar_cos.flatten()
    flatten_cos.sort()

    response = '{}'
    if (flatten_cos[-2] == 0):
        response = response.format('I am sorry, I didn\'t get that!')
    else:
        response = response.format(sents[similar_idx])
    return response


def start_chatbot_session(sents: list, intents: set, stop_flag='quit'):
    print('Ask chatbot related questions!\n'
          f'enter: < {stop_flag} > to stop anytime.\n')

    while True:
        chatbot_reply = 'bot : {}'
        text_input = input('you : ').lstrip().lower()
        if text_input != stop_flag:
            if text_input in intents:
                print(chatbot_reply.format(
                    greeting_response(text_input, intents=intents)
                ))
            else:
                print(chatbot_reply.format(
                    similar_response(text_input, sents=sents)
                ))
        elif text_input == stop_flag:
            print(chatbot_reply.format('Bye, take care!'))
            break
        else:
            continue


raw_docs = get_textdocs('chatbot.txt')
sent_tokens = sentence_tokenizer(raw_docs)
start_chatbot_session(sents=sent_tokens,
                      intents=GREET_INTENTS,
                      stop_flag='bye')
