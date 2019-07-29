
from nltk import sent_tokenize
from nltk import word_tokenize
import numpy as np


class TextLabeler():
    def __init__(self, text, dict_list):
        '''Assigns a label format to a given key value pair.

        text : (str)
            The text that contains the values to replace.

        dict_list : (list)
         A list of dict key value pairs [{'key': 'value'}]

         >>> dict_list = [{'city':'Paris'}]
         >>> text = "A trip to Paris"
         >>> label_keys = TextLabeler(text, dict_list)
         > "A trip to [Paris](city) : [value](key)"
        '''
        self.text = text
        self.iterate(dict_list)

    def replace_key_value(self, label):
        '''Replace any occurrence of a value with the key
        '''
        for key, value in label.items():
            label = """[{1}]({0})""".format(key, value)
            self.text = self.text.replace(value, label)
            return self.text

    def iterate(self, dict_list):
        '''Iterate over each dict object in a given list of dicts, `dict_list`
        '''
        for label in dict_list:
            self.text = self.replace_key_value(label)
        return self.text


class ReplaceKeyValues():

    def __init__(self, text, dict_list):
        self.text = text
        self.iterate(dict_list)

    def replace_key_value(self, label):
        '''Replace any occurrence of a value with the key
        '''
        for key, value in label.items():
            self.text = self.text.replace(value, key)
            return self.text

    def iterate(self, dict_list):
        '''Iterate over each dict object in a given list of dicts, `dict_list`
        '''
        for label in dict_list:
            self.text = self.replace_key_value(label)
        return self.text


def create_batch(self, sentence_list):
    """Create a batch for a list of sentences
    """
    embeddings_batch = []

    for sent in sentence_list:
        embeddings = []
        sent_tokens = sent_tokenize(sent)
        word_tokens = [word_tokenize(w) for w in sent_tokens]
        tokens = [value for sublist in word_tokens for value in sublist]
        tokens = [token for token in tokens if token != '']

        for token in tokens:
            embeddings.append(self.embdict.tok2emb.get(token))

        if len(tokens) < self.max_sequence_length:
            pads = [np.zeros(self.embedding_dim) for _ in range(
                self.max_sequence_length - len(tokens)
            )]
            embeddings = pads + embeddings
        else:
            embeddings = embeddings[-self.max_sequence_length:]

        embeddings = np.asarray(embeddings)
        embeddings_batch.append(embeddings)

    embeddings_batch = np.asarray(embeddings_batch)
    return embeddings_batch


class SchoolMember:
    '''Represents any school member.'''

    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('(Initialized SchoolMember: {})'.format(self.name))

    def tell(self):
        '''Tell my details.'''
        print('Name:{} Age:{}'.format(self.name, self.age), end=' ')


class Teacher(SchoolMember):
    '''Represents a teacher.'''

    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print('(Initialized Teacher: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: {:d}'.format(self.salary))


class Student(SchoolMember):
    '''Represents a student.'''

    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('(Initialized Student: {})'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Marks: {:d}'.format(self.marks))


t = Teacher('Mrs. Shrividya', 40, 30000)
s = Student('Swaroop', 25, 75)
# prints a blank line
print()
members = [t, s]
for member in members:
    # Works for both Teachers and Students
    member.tell()
