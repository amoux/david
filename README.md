# david nlp 💬

* David is an NLP toolkit implemented with Gensim, Tensorflow, PyTorch, NLTK, and spaCy among other open-source libraries.

![image](https://fromdirectorstevenspielberg.com/wp-content/uploads/2017/07/15.jpg?raw=true)

*The goal for David is to assist content creators to increase the exposure on YouTube and their videos in the presence of more viewers. From live/historical textual information.*

## configuration

* clone or download the repo. use `git pull` to have the latest release.

```bash
git clone https://github.com/amoux/david
```

Error: ***OSError: mysql_config not found***

> **NOTE** before you install all the dependecies, the package `pattern` is known to have a common error due to the missing `libmysqlclient-dev` linux package. to fix this issue, execute: `sudo apt install libmysqlclient-dev` and then follow the steps below.

* after installing the missing package - in a virtual enviroment then run:

```bash
git clone https://github.com/clips/pattern
cd pattern/
python setup.py install
```

### requirements

* CPU

```bash
pip install -r cpu-requirements.txt
```

* GPU

```bash
pip install -r gpu-requirements.txt
```

> to install everything (including all dependancies) run:

* navigate to the repo's root directory and install.

```bash
pip install .
```

> **NOTE** download the required language models with one command (you don't need to be in the root project directory).

* the following models will be downloaded.
  * `en_core_web_sm`
  * `en_core_web_lg`

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

* configure the database and build a dataset from a search query. default parameters `db_name='comments_v2.db', table='comments'`.

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

* export a document to a df with the `export` attribute.

```python
from david.pipeline import Pipeline

pipe = Pipeline(comments.export('df'))
```

* the following metrics are available one call away 🤖

```python
pipe.get_all_metrics(string=True, words=True, characters=True, tags=True)
pipe.describe()
```

* with `tags=True` the following attributes are available. (the amount of tags varies on the size of the dataset)

```ipython
pipe.authorEmoji.unique()
...
array(['👍', '😍😍', '😂💙👄', '😊', '💕💕💕', '✌🏾', '😙', '🤔🤷♂'],
dtype=object)

pipe.authorUrlLink.unique()
...
array([nan, 'https://www.youtube.com/channel/UCywXyzx6GZpDyxRvMOqLMiw'],
dtype=object)

pipe.authorHashTags.unique()
...
array([nan, '#SUBSCRIBED'], dtype=object)

pipe.authorTimeTag.unique()
...
array([nan, '10:06'], dtype=object)
```

## preprocessing 🔬

* text cleaning routines.

### stop words

> you can access multiple collections of stop words from the lang module. all the following are available: `DAVID_STOP_WORDS, GENSIM_STOP_WORDS, NLTK_STOP_WORDS, SPACY_STOP_WORDS`

```python
from david.lang import SPACY_STOP_WORDS

# if stop_words param left as None, it defaults to spaCy's set.
pipe_stop_words = pipe.custom_stopwords_from_freq(
       top_n=30, stop_words=SPACY_STOP_WORDS)
list(pipe_stop_words)[:5]
```

* returns a set containing the most frequent words used in a dataset and adds them to any existing collection of stop-words (in this case we have top words from both our corpus and spaCy's set)

```ipython
['tides...cardinal', 'into', 'less', 'same', 'under']
```
