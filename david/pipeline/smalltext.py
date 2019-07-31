import re
import os
import unicodedata

# inflect: correctly generate plurals, singular
# nouns, ordinals, indefinite articles
import inflect
import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


class Text:

    def _datasets(self):
        resource = os.path.abspath()

    def _html_textparser(self, text: str):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def _html_betweenbrackets(self, text: str):
        # remove between square brackets
        return re.sub("\[[^]*\]", "", text)

    def strip_htmlfromtext(self, text: str):
        text = self._html_textparser(text)
        text = self._html_betweenbrackets(text)
        return text

    def replace_contractions(self, text: str, leftovers=True, slang=True):
        return contractions.fix(text, leftovers, slang)

    def remove_non_ascii(words: list):
        '''Remove non-ASCII characters from list of tokenized words.
        '''
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode(
                'ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words: list):
        '''Convert all characters to lowercase from
        list of tokenized words.
        '''
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words: list):
        '''Remove punctuation from list of tokenized words
        '''
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words: list):
        '''Replace all interger occurrences in list of
        tokenized words with textual representation
        '''
        engine = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = engine.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words: list):
        '''Remove stop words from list of tokenized words.
        '''
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words

    def stem_words(self, words: list):
        '''Stem words in list of tokenized words.
        '''
        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words: list):
        '''Lemmatize verbs in list of tokenized words
        '''
        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def stem_and_lemmatize(self, words: list):
        '''RETURNS: stems, lemmas = stem_and_lemmatize(words)
        '''
        stems = self.stem_words(words)
        lemmas = self.lemmatize_verbs(words)
        return stems, lemmas

    def preprocess(
        self,
        texts: list,
        denoise=False,
        rm_contractions=False,
        rm_non_ascii=True,
        lower_text=True,
        rm_punctuation=True,
        replace_nums=False,
        rm_stopwords=True,
        lemm=None,
        stemm=None,
    ):
        if strip_htmlfromtext:
            texts = [x for x in self.denoise(texts)]
        if rm_contractions:
            texts = [x for x in self.replace_contractions(texts)]
        if rm_non_ascii:
            texts = self.remove_non_ascii(texts)
        if lower_text:
            texts = self.to_lowercase(texts)
        if rm_punctuation:
            texts = self.remove_punctuation(texts)
        if replace_nums:
            texts = self.replace_numbers(texts)
        if rm_stopwords:
            texts = self.remove_stopwords(texts)

        if lemm and not stemm:
            texts = self.lemmatize_verbs(texts)
        if stemm and not lemm:
            texts = self.stem_words(texts)
        return texts


def listfiles_in_directorytree(dirname: str):
    '''For the given path, get the List of all files
    in the directory tree.

    >>> `add` : 'return a list of files and sub-directories'
    '''
    # create a list of files and sub-directories names in the given directory.
    entries = os.listdir(dirname)
    # iterate over all the entries
    allfiles = list()
    for entry in entries:
        # create full path
        fullpath = os.path.join(dirname, entry)
        # if entry is a directory then get the list of files in this directory
        if os.path.isdir(fullpath):
            allfiles = allfiles + listfiles_in_directorytree(fullpath)
        else:
            allfiles.append(fullpath)


def getpaths_by_filetype(dirname: str, file_endswith='.json'):
    '''Get the path for any specific file type.
    file_endswith : e.g. `.txt, .csv`
    '''
    filepaths = []
    allfiles = getpaths_by_filetype(dirname)
    for file in allfiles:
        if file.endswith(file_endswith):
            filepaths.append(file)
    return filepaths


# def _html_textparser(text: str):
#     soup = BeautifulSoup(text, "html.parser")
#     return soup.get_text()


# def _html_betweenbrackets(text: str):
#     return re.sub('\[[^]]*\]', '', text)


# def strip_htmlfromtext(text: str):
#     text = _html_textparser(text)
#     text = _html_betweenbrackets(text)
#     return text


# def replace_contractions(text: str):
#     '''Replace contractions in string of text
#     '''
#     return contractions.fix(text)


# def remove_non_ascii(words: list):
#     '''Remove non-ASCII characters from list of tokenized words
#     '''
#     new_words = []
#     for word in words:
#         new_word = unicodedata.normalize('NFKD', word).encode(
#             'ascii', 'ignore').decode('utf-8', 'ignore')
#         new_words.append(new_word)
#     return new_words


# def to_lowercase(words: list):
#     '''
#     Convert all characters to lowercase from
#     list of tokenized words
#     '''
#     new_words = []
#     for word in words:
#         new_word = word.lower()
#         new_words.append(new_word)
#     return new_words


# def remove_punctuation(words: list):
#     '''Remove punctuation from list of tokenized words
#     '''
#     new_words = []
#     for word in words:
#         new_word = re.sub(r'[^\w\s]', '', word)
#         if new_word != '':
#             new_words.append(new_word)
#     return new_words


# def replace_numbers(words: list):
#     '''
#     Replace all interger occurrences in list of
#     tokenized words with textual representation
#     '''
#     p = inflect.engine()
#     new_words = []
#     for word in words:
#         if word.isdigit():
#             new_word = p.number_to_words(word)
#             new_words.append(new_word)
#         else:
#             new_words.append(word)
#     return new_words


# def remove_stopwords(words: list):
#     '''Remove stop words from list of tokenized words
#     '''
#     new_words = []
#     for word in words:
#         if word not in stopwords.words('english'):
#             new_words.append(word)
#     return new_words


# def stem_words(words: list):
#     '''Stem words in list of tokenized words
#     '''
#     stemmer = LancasterStemmer()
#     stems = []
#     for word in words:
#         stem = stemmer.stem(word)
#         stems.append(stem)
#     return stems


# def lemmatize_verbs(words: list):
#     '''Lemmatize verbs in list of tokenized words
#     '''
#     lemmatizer = WordNetLemmatizer()
#     lemmas = []
#     for word in words:
#         lemma = lemmatizer.lemmatize(word, pos='v')
#         lemmas.append(lemma)
#     return lemmas


# def normalize(words: list):
#     words = remove_non_ascii(words)
#     words = to_lowercase(words)
#     words = remove_punctuation(words)
#     words = replace_numbers(words)
#     words = remove_stopwords(words)
#     return words


# def stem_and_lemmatize(words: list):
#     '''
#     RETURNS:
#     stems, lemmas = stem_and_lemmatize(words)
#     '''
#     stems = stem_words(words)
#     lemmas = lemmatize_verbs(words)
#     return stems, lemmas
