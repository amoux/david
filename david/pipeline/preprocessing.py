import re as _re
import string as _string
import contractions as _contractions
from itertools import groupby
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer as _WordNetLemmatizer


class TextPreprocess(pd.DataFrame):
    def __init__(self, file_path):
        super().__init__(pd.read_json(file_path, encoding='utf-8', lines=True))

    def to_textfile(self, file_name, text_col='text'):
        '''Save the texts from a column to a text file and
        skips any line with == 0 value.
        '''
        with open(file_name, 'w', encoding='utf-8') as txt:
            lines = self[text_col].tolist()
            for line in lines:
                if len(line) != 0:
                    txt.write('%s\n' % line)
            txt.close()

    def missing_values(self):
        return self.isnull().sum()

    def remove_whitespaces(self, text: str):
        '''Remove more than one space.
        '''
        return _re.sub(r'(\s)\1{1,}', ' ', text).strip()

    def tokenize(self, texts: list):
        '''Return the tokens of a sentence
        including punctuation.
            >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob','dropped','the','apple', '.','Where','is','the','apple','?']
        '''
        return [x.strip() for x in _re.split(r'(\W+)?', texts) if x.strip()]

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
        word_map = text.maketrans(dict.fromkeys(_string.punctuation))
        word_clean = text.translate(word_map)
        return ' '.join([k for k, v in groupby(word_clean.split())])

    def reduce_repeatingchars(self, text: str):
        '''Reduces repeated `characters`
        '''
        findings = _re.findall(r'(\w)\1{2,}', text)
        for char in findings:
            find = char + '{3,}'
            replace = '???' + char + '???'
            text = _re.sub(find, repr(replace), text)
        text = text.replace('\'???', '')
        text = text.replace('???\'', '')
        text = self.remove_whitespaces(text)
        return text

    def replace_contractions(self, text_col='text'):
        '''Replaces contractions (including slang words)
        NOTE: This process requires normal spacing between characters.
        '''
        self[text_col] = self[text_col].apply(lambda x: _contractions.fix(x))

    def _TextLemmatizer(self, texts: list):
        WordNet = _WordNetLemmatizer()
        return ' '.join([WordNet.lemmatize(w) for w in texts])

    def normalize_whitespaces(self, text_col='text'):
        '''Prep texts by normalizing whitespaces.'''
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
            lambda x: self._TextLemmatizer(x))

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

    def clean_all_text(self, text_col='text', standardize=True,
                       contractions=True, lemmatize=False, normalize=True,
                       rm_duplicates=True, lower_text=False):
        '''Dataframe Text Preprocessing Method. The arrangements
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
            self.lower_text()
