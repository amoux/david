# david nlp 💬

> The goal of this toolkit is to speed the time-consuming steps to obtain, store, and pre-process textual data from youtube, and implementing natural language processing techniques to extract highly specific information, such as the indications of product or service trends across enterprises. There's much potential in being able to get valuable information for analytics.

- 👨‍💻 In progress ( Web-App )

  - Vuepoint is an analytics web-application that helps content creators by automating the tedious task of manually having to scroll and read through multiple pages of comments to understand what an audience wants. Extracting the vital information that is relevant to each content creator without the noise!

- 📃 objectives:

  - Semantic analytics application for content creators.
  - Social marketing trends of interest.

- 🤔 **TODO**
  - Update Readme documentation on recently added/updated classes and methods.
  - Improve text preprocessing methods (covert all to Generators).
  - Improve pipeline customization (when switching from Pipeline to a custom recipe).
  - Update toolkit with `PEP 484` - Support for type hints.

## configuration 👻

- First clone or download the repo. use `git pull` to have the latest release.

```bash
git clone https://github.com/amoux/david
```

### requirements

- Install the requirements and install the package:

```bash
pip install -r requirements.txt
pip install
```

### spaCy language models

> Download the required language models with one command (you don't need to be in the root project directory).

- The following models will be downloaded.
  - `en_core_web_sm`
  - `en_core_web_lg`

```bash
$ download-spacy-models
...
✔ Download and installation successful
You can now load the model via spacy.load('en_core_web_lg')
...
✔ Download and installation successful
You can now load the model via spacy.load('en_core_web_sm')
```

## server 📡

- Configure the database and build a dataset from a search query. Using an existing database of youtube comments - here we are use the `unbox` database (The database will be downloaded automatically if it doesn't exist in the `david_data` directory).

```python
from david.server import CommentsSql
db = CommentsSql('unbox')

# Fetch a batch based on a query.
query = "%make a video%"
columns = "id, cid, text"
batch = db.fetch_comments(query, columns, sort_col='id')

# Access the batch columns by simply passing the index of the batch:
idx, cid, text = batch[10]
print(text)
...
'Hey dude quick question when are u gonna make a video with the best phone of 2018?'
```

> Building a simple classification dataset from youtube comments.

```python
question = 1
statement = 0

# dataset with 100 samples.
dataset = []
for i in range(100):
    text = batch[i].text.strip()
    if text is not None:
      label = question if text.endswith('?') else statement
      dataset.append((text, label))
dataset[:10]
...
 [('Try the samsung a9 and make a video like "the smartphone with 4 cameras"', 0),
 ("Yo you're going to make a video on the pixel Slate bro?", 1),
 ('Would you please make a video on Funcl W1 and Funcl AI earphones.', 0), ...]
```

## tokenizers

> New tokenizers added in the text module. While the are functional the module is still in progress. The both the `CharacterTokenizer` and `SentenceTokenizer` classes inherent the `VocabularyBase` class.

TODO: Add documentation and more examples for using these classes.

```python
from david.text.tokenizers import (CharacterTokenizer,
                                   WordTokenizer, SentenceTokenizer)
char_tokenizer = CharacterTokenizer()  # character level tokenizer.
sm_tokenizer = WordTokenizer()  # word level tokenizer.
sent_tokenizer = SentenceTokenizer()  # sentence level tokenizer.

# Perfect for tokenizing social media text.
sm_tokenizer.tokenize("This is a 👻 emoji #character, and @david is getting new updates!")
...
['This', 'is', 'a', '👻', 'emoji', '#character', ',',
'and', '@david', 'is', 'getting', 'new', 'updates', '!']

# This is another functionality that will be included in all tokenizers
# In the meantime, only the character tokenizer has this feature. Here we
# embed (torch.TensorType) one single character.
char_embeddings = char_tokenizer.character_to_tensor("A")
...
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0.]])

# Finally lets tokenize a document of sequences:
sentences = ("There are the new tokenizers available with david! "
    "The WordTokenizer properly handles all your emoji's ect. "
    "The CharacterTokenizer works with a single character (for word embeddings)"
    "The SentenceTokenizer properly converts chunks of texts into sentences!")

# Tokenized sentences.
tokenized_sents = sent_tokenizer.tokenize(sentences)
...
['<start> there are the new tokenizers available with david ! <end>',
'<start> the wordtokenizer properly handles all your emoji s ect . <end>',
'<start> the charactertokenizer works with a single character for word embeddings'
'the sentencetokenizer properly converts chunks of texts into sentences ! <end>']
```

## pipeline 🛠

- export a document to a df with the `export` attribute.

```python
from david.pipeline import Pipeline
pipe = Pipeline(dataset, columns=['text', 'label'])
pipe.head()
...
                                                text  label
0  Can you make a video about pixel 1 2 and 3 sma...      1
1  the meme is not because you change your phone,...      1
2                   Plzz us note 9 and  make a video      1
3  Lewis Please Make A Video On S10 5G Model I Th...      1
4  Hey lew\nCan you please make a video on huawei...      1
```

> The following metrics are available one call away 🤖

```python
pipe.get_all_metrics(
  string=True,
  words=True,
  characters=True,
  tags=True,
)
pipe.describe()
```

- With `tags=True` the following attributes are available. (the amount of tags varies on the size of the dataset)

```ipython
pipe.authorEmoji.unique()
...
array(['👍', '😍😍', '😂💙👄', '😊', '💕💕💕', '✌🏾', '😙', '🤔🤷♂'],
dtype=object)
```

## preprocessing 🔬

- Text cleaning routines.

### stop words

> You can access multiple collections of stop words from the lang module. all the following are available: `DAVID_STOP_WORDS, GENSIM_STOP_WORDS, NLTK_STOP_WORDS, SPACY_STOP_WORDS`

```python
from david.lang import SPACY_STOP_WORDS
# if stop_words param left as None, it defaults to spaCy's set.
pipe_stop_words = pipe.custom_stopwords_from_freq(top_n=30, stop_words=SPACY_STOP_WORDS)
list(pipe_stop_words)[:5]
```

- Returns a set containing the most frequent words used in a dataset and adds them to any existing collection of stop-words (in this case we have top words from both our corpus and spaCy's set)

```ipython
['tidees...cardinal', 'into', 'less', 'same', 'under']
```

- A quick look at the results from the three possible preprocessing modes.

```python
from david.text import preprocess_docs
doc_a = preprocess_docs(docs, stopwords=False, tokenize=True)
doc_b = preprocess_docs(docs, stopwords=True, tokenize=True)
doc_c = preprocess_docs(docs, stopwords=False, tokenize=False)
doc_d = preprocess_docs(docs, stopwords=True, tokenize=True, lemma=True)
```

```python
doc_a[:3] # stopwords=False, tokenize=True
...
[['Love', 'it'],
 ['Put', 'the', 'solar', 'kit', 'on', 'top', 'during', 'the', 'day'],
 ['Police', 'car', 'runs', 'out', 'of', 'gas', 'during', 'chase']]

doc_b[:3] # stopwords=True, tokenize=True
...
 [['Love'],
 ['Put', 'solar', 'kit', 'top', 'day'],
 ['Police', 'car', 'runs', 'gas', 'chase']]

doc_c[:3] # stopwords=False, tokenize=False
...
 ['Love it',
  'Put the solar kit on top during the day',
  'Police car runs out of gas during chase']
```

- Applying lemma vs not

```python
doc_b[5:6] # lemma not applied
...
[['Except', 'filling', 'gas', 'takes', 'like', '10', 'minutes',
  'charging', 'Tesla', 'takes', 'several', 'hours']]

doc_d[5:6] # lemma applied
...
[['except', 'fill', 'gas', 'take', 'like', '10', 'minute',
'charge', 'tesla', 'take', 'several', 'hour']]
```

### general info

- Default linter style settings:

```code
{
  "python.linting.pycodestyleEnabled": true,
  "python.linting.enabled": true,
  "python.linting.pycodestyleArgs": ["--ignore=E402"],
}
```
