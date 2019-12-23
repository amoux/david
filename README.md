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

- clone or download the repo. use `git pull` to have the latest release.

```bash
git clone https://github.com/amoux/david
```

Error installing `pattern`: **_OSError: mysql_config not found_**

> **NOTE** before you install all the dependencies, the package `pattern` is known to have a common error due to the missing `libmysqlclient-dev` linux package. to fix this issue run the following command before installing the requirements.

```bash
sudo apt install libmysqlclient-dev
conda activate < YOUR ENVIRONMENT >
(conda-env) cd pattern/
(conda-env) python setup.py install
```

- If you still get `fatal error: mysql/udf_registration_types.h: No such file or directory #include <mysql/udf_registration_types.h>` when installing in a Ubuntu + Anaconda environment try:

- tested on: Ubuntu 19.10
- anaconda environment: python=3.6

```bash
(conda-env) conda install gxx_linux-64
(conda-env) conda install mysqlclient
(conda-env) cd pattern/
(conda-env) python setup.py install
```

### requirements

- install the requirements:

```bash
pip install -r requirements.txt
```

- in the root directory then install the package:

```bash
pip install .
```

> **NOTE** download the required language models with one command (you don't need to be in the root project directory).

- the following models will be downloaded.
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

- configure the database and build a dataset from a search query. default parameters `db_name='comments_v2.db', table='comments'`.

```python
from david.server import CommentsDB

db = CommentsDB()
comments = db.get_all_comments()
[c.text for c in comments][:5]
...
['Video was hilarious, subscribed!',
 'Hamazingly educational; Subscribed for more :)',
 'Great vid....SUBSCRIBED with the Bell👍',
 'Clicked for the learning, stayed for the hair,
 subscribed for the humor. Keep it up fam.',
 'I keep thinking the slinky on his head is going
 to fall off. But this stuff is too damn
 interesting so I subscribed anyways.']
```

## pipeline 🛠

- export a document to a df with the `export` attribute.

```python
from david.pipeline import Pipeline
pipe = Pipeline(comments.export('df'))
```

- the following metrics are available one call away 🤖

```python
pipe.get_all_metrics(string=True, words=True, characters=True, tags=True)
pipe.describe()
```

- with `tags=True` the following attributes are available. (the amount of tags varies on the size of the dataset)

```ipython
pipe.authorEmoji.unique()
...
array(['👍', '😍😍', '😂💙👄', '😊', '💕💕💕', '✌🏾', '😙', '🤔🤷♂'],
dtype=object)
```

## preprocessing 🔬

- text cleaning routines.

### stop words

> you can access multiple collections of stop words from the lang module. all the following are available: `DAVID_STOP_WORDS, GENSIM_STOP_WORDS, NLTK_STOP_WORDS, SPACY_STOP_WORDS`

```python
from david.lang import SPACY_STOP_WORDS
# if stop_words param left as None, it defaults to spaCy's set.
pipe_stop_words = pipe.custom_stopwords_from_freq(top_n=30, stop_words=SPACY_STOP_WORDS)
list(pipe_stop_words)[:5]
```

- returns a set containing the most frequent words used in a dataset and adds them to any existing collection of stop-words (in this case we have top words from both our corpus and spaCy's set)

```ipython
['tides...cardinal', 'into', 'less', 'same', 'under']
```

- a quick look at the results from the three possible preprocessing modes.

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

- applying lemma vs not

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
