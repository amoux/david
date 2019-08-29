import re
import string
from itertools import groupby

import contractions
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer


class TextPreprocess(pd.DataFrame):
    JSON_PATH = None

    def __init__(self, json_fpath):
        super().__init__(pd.read_json(
            json_fpath, encoding='utf-8', lines=True))
        self.JSON_PATH = json_fpath

    def to_textfile(self, fn, text_col='text'):
        '''Save the texts from a column to a text file and
        skips any line with == 0 value.
        '''
        with open(fn, 'w', encoding='utf-8') as f:
            for x in self[text_col].tolist():
                if len(x) != 0:
                    f.write('%s\n' % x)
            f.close()

    def get_json_filepath(self):
        '''Returns the path of the corpus for other instances.
        NOTE: if this works better than having to make another class
        just to swap an object from one instance to another. just
        delete this NOTE and get a coffee. Simple IS BETTER!
        '''
        return self.JSON_PATH

    def obj_todict(self, df_obj: object):
        return df_obj.to_dict(orient='index')

    def missing_values(self):
        return self.isnull().sum()

    def remove_whitespaces(self, text: str):
        '''Remove more than one space.
        '''
        return re.sub(r'(\s)\1{1,}', ' ', text).strip()

    def tokenize(self, texts: list):
        '''Return the tokens of a sentence
        including punctuation.
            >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob','dropped','the','apple', '.','Where','is','the','apple','?']
        '''
        return [x.strip() for x in re.split(r'(\W+)?', texts) if x.strip()]

    def remove_duplicatewords(self, text: str):
        '''Recommended use after preprocessing texts. For better results
        `Lemmatization` is one way to further improve word matches.
        Returns strings with no repeated words in sequence.

        Example ::
            >>> text = "Hey! you are wrong very wrong! wrong!"
            >>> text = remove_duplicatewords(text)
            >>> "Hey you are wrong very wrong"
        '''
        # remove punctuation
        word_map = text.maketrans(dict.fromkeys(string.punctuation))
        word_clean = text.translate(word_map)
        return ' '.join([k for k, v in groupby(word_clean.split())])

    def reduce_repeatingchars(self, text: str):
        '''Reduces repeated `characters`
        '''
        findings = re.findall(r'(\w)\1{2,}', text)
        for char in findings:
            find = char + '{3,}'
            replace = '???' + char + '???'
            text = re.sub(find, repr(replace), text)
        text = text.replace('\'???', '')
        text = text.replace('???\'', '')
        text = self.remove_whitespaces(text)
        return text

    def replace_contractions(self, text_col='text'):
        '''
        Replaces contractions (including slang words)
        NOTE: This process requires normal spacing between characters.
        '''
        self[text_col] = self[text_col].apply(lambda x: contractions.fix(x))

    def lemmatizer(self, texts: list):
        WordNet = WordNetLemmatizer()
        return ' '.join([WordNet.lemmatize(w) for w in texts])

    def normalize_whitespaces(self, text_col='text'):
        '''Prep texts by normalizing whitespaces.
        '''
        self[text_col] = self[text_col].str.strip()

    def lower_text(self, text_col='text'):
        self[text_col] = self[text_col].str.lower()

    def standardize_text(self, text_col='text'):
        self[text_col] = self[text_col].str.replace(r" '", r"'")
        self[text_col] = self[text_col].str.replace(r"http\S+", "")
        self[text_col] = self[text_col].str.replace(r"http", "")
        self[text_col] = self[text_col].str.replace(r"@\S+", "")
        self[text_col] = self[text_col].str.replace(
            r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")

    def lemmetize_texts(self, text_col='text'):
        self[text_col] = self[text_col].str.split()
        self[text_col] = self[text_col].apply(
            lambda x: self.lemmatizer(x))

    def normalize_texts(self, text_col='text'):
        self[text_col] = self[text_col].str.replace(
            r"&lt;/?.*?&gt;", " &lt;&gt; ")
        self[text_col] = self[text_col].str.replace(r"(\\d|\\W)+", " ")
        self[text_col] = self[text_col].str.replace(r"[^a-zA-Z]", " ")

    def remove_duplicatewords_andchars(self, text_col='text'):
        self[text_col] = self[text_col].apply(
            lambda x: self.remove_duplicatewords(x))
        self[text_col] = self[text_col].apply(
            lambda x: self.reduce_repeatingchars(x))

    @staticmethod
    def process_fromchunks(self, infile='', sep=',', chunksize=1000,
                           outfile='df_output.csv', text_col='text',
                           standardize=True, contractions=True,
                           lemmatize=False, normalize=True,
                           rm_duplicates=True, lower_text=False):
        '''
        NOTE: This function needs to be tested and optimized.
        Chunking is importantly usefull for any large corpus so its
        not another feature. Its a must have feature here!
        For more info on this features see the notes and tests i took
        for this function on a jupyter notebook! DONT FORGET!

        Jupyter-Notebooks/David/02_pipe/chunking-testing-spacy-language-detection.ipynb
        '''

        df_chunks = pd.read_csv(infile, sep=sep, chunksize=chunksize)
        tempchunks = []
        for df in df_chunks:
            self.clean_all_text(df, text_col, standardize,
                                contractions, lemmatize, normalize,
                                rm_duplicates, lower_text)
        return pd.concat(tempchunks)

    def clean_all_text(self, text_col='text', standardize=True,
                       contractions=True, lemmatize=False, normalize=True,
                       rm_duplicates=True, lower_text=False):
        '''
        Dataframe Text Preprocessing Method. The arrangements
        have been crafted uniquely for Youtube Comments. Though,
        It can process similar styles from social-media text;
        (e.g. tweets or Reddit-posts). Cleaning text is recommended for
        increasing the accuracy/results from NLP models (e.g.
        `word2vec`, & `Gensim's LDA`).

        PARAMETERS
        ----------

        `text_col` : (str, df['dataframe column'])
            Pass the name of the column where the text is located.

        `lowertext` : (bool, default=True)
            Converts all texts to lowercase.

        `contractions` : (bool, default=True)
            Replaces common contractions (`including slang words`)

        `lemmatize` : (bool, default=True)
            Lemmatizes words: Useful for models like `LDA` and `word2vec`.

        `rm_duplicates` : (bool, default=True)
            Removes duplicate words (`words next-to-eachother ONLY`).
            It also reduces words with chars repeated more than 3 times
            to a single char. Useful to replace words such as `looooooong`
            by `long`. `NOTE`: It can change abbreviations such as `AAA`
            to single `A`

                Example:
                >>> text = "Hey! you are wrong very wrong! wrong!"
                >>> text = remove_duplicatewords_andchars(text)
                >>> "Hey you are wrong very wrong"

        ABOUT-NORMALIZER
        ----------------
        * Normalizes corpus by trimming whitespaces,
        removes all numbers urls, hashtags, mention
        tags found in the columns.
        '''
        self.normalize_whitespaces(text_col)
        if contractions:
            self.replace_contractions(text_col)
        if standardize:
            self.standardize_text(text_col)
        if lemmatize:
            self.lemmetize_texts(text_col)
        if normalize:
            self.normalize_texts(text_col)
        if rm_duplicates:
            self.remove_duplicatewords_andchars(text_col)
        if lower_text:
            self.lower_text(text_col)
