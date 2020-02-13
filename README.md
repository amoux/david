# david nlp  💬 (social-media-toolkit)

The goal of this toolkit is to speed the time-consuming steps to obtain, store, and pre-process textual data from youtube, and implementing natural language processing techniques to extract highly specific information, such as the indications of product or service trends across enterprises. There's much potential in being able to get valuable information for analytics.

- Projects using this toolkit:

  - **vuepoint** (flask-web-app)
    - Vuepoint is an analytics web-application that helps content creators by automating the tedious task of manually having to scroll and read through multiple pages of comments to understand what an audience wants. Extracting the vital information that is relevant to each content creator without the noise!

  - **david-sentiment** (embedding-models):
    - Train an accurate sentiment model based on `meta-learning` and `embedding` techniques with small datasets and a few lines of code.

  - **qaam-nlp** (question-answering): 
    - Given an article or blog URL with text content. The model (based on the BERT) will answer questions on the given context or subject.

- 📃 Objectives:
  - Build a rich toolkit for **end-to-end** *NLP* pipelines
  - Specialized text preprocessing techniques for social-media text.
  - Social-Datasets: Currently working only on ***YouTube***, but then plan to extend to similar sites likes *Reddit*, and *Twitter* in the future.

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

### spaCy models

Download the required language models with one command (you don't need to be in the root project directory).

> The following models will be downloaded: ***en_core_web_sm*** ++ ***en_core_web_lg***

```bash
$ download-spacy-models
...
✔ Download and installation successful
You can now load the model via spacy.load('en_core_web_lg')
...
✔ Download and installation successful
You can now load the model via spacy.load('en_core_web_sm')
```

## `david.tokenizers.Tokenizer`

The base class `david.tokenizers.BaseTokenizer` implements the common methods for loading/saving a tokenizer either from a local file or director. Support for downloading tokenizer models will be added in the future.

- tokenizing, converting tokens to ids and back and encoding/decoding,
- adding new tokens to the vocabulary in a way that is independent of the underlying structure.
  
For this demo, 1600 samples works but you can choose up to 6k samples.

```python
from david.datasets import YTCommentsDataset
train_data, _ = YTCommentsDataset.split_train_test(2000, subset=0.8)
print('text:', train_dataset[0])
print('size:', len(train_dataset))

* text: 'This is very Good Way to Wake up myself from dreaming Fairy Life. Feeling Energetic Now.'
* size: 1600
```

Construct a Tokenizer object and pass a document to build the vocabulary.

```python
from david.tokenizers import Tokenizer
tokenizer = Tokenizer(document=train_data,
                      remove_urls=True,
                      enforce_ascii=True,
                      reduce_length=True,
                      preserve_case=False,
                      strip_handles=False)
print(tokenizer)
'<Tokenizer(vocab_size=7673)>'
```

Align the vocabulary index relative to its term frequency.

```python
tokenizer.index_vocab_to_frequency(inplace=True)
```

- Before alignment

```python
* vocab_index: [('this', 1), ('is', 2), ('very', 3), ('good', 4), ('way', 5)]
* vocab_count: [('.', 2215), ('the', 2102), (',', 1613), ('to', 1286), ('i', 1277)]
```

- After alignment

```python
* vocab_index: [('.', 1), ('the', 2), (',', 3), ('i', 4), ('to', 5)]
* vocab_count: [('.', 2215), ('the', 2102), (',', 1613), ('i', 1297), ('to', 1286)]
```

## `david.server.CommentsSql` 📡

- Configure the database and build a dataset from a search query. Using an existing database of youtube comments - here we are use the `unbox` database (The database will be downloaded automatically if it doesn't exist in the `david_data` directory).

```python
from david.server import CommentsSql
 # loading an existing database.
db = CommentsSql('unbox')
batch = db.fetch_comments(query="%make a video%",
                          columns="id,cid,text",
                          sort_col='id')
idx, cid, text = batch[10]
print(text)
...
'Hey dude quick question when are u gonna make a video with the best phone of 2018?'
```

Example of 'building' a dataset the batch:

```python
# dataset with 100 samples.
question = 1
statement = 0
dataset = []
for i in range(100):
    text = batch[i].text.strip()
    if text is not None:
      label = question if text.endswith('?') else statement
      dataset.append((text, label))

print(dataset[:3])
...
 [('Try the samsung a9 and make a video like "the smartphone with 4 cameras"', 0),
 ("Yo you're going to make a video on the pixel Slate bro?", 1),
 ('Would you please make a video on Funcl W1 and Funcl AI earphones.', 0), ...]
```

## david.text 🔬

A quick look at the results from the three possible preprocessing modes.

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
