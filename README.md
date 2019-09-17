# david - Vuepoint Analytics

* David is an NLP toolkit implemented with Gensim, Tensorflow, PyTorch, NLTK, and spaCy among other open-source libraries. At Vuepoint, we are applying models built on these libraries for text-classification, intent-detection, topic-modeling to improve youtube content creators view of their audience to produce more suitable content.

![image](https://fromdirectorstevenspielberg.com/wp-content/uploads/2017/07/15.jpg?raw=true)

*The goal for David is to assist content creators to increase the exposure on YouTube and their videos in the presence of more viewers. From live/historical textual information.*

> Currently the project is still in beta stages. Once the library is stable enough, an enhanced realease will be available for the public to use.

* build a dataset from a search query.

```python
>>> from vuepoint.david_server.sql import SqliteCommentsDB
>>> from vuepoint.david.pipeline import Pipeline
...
>>> docs = sql.get_similartexts('i subscribed')
>>> [doc.text for doc in docs][:5]

'['Video was hilarious, subscribed!',
 'Hamazingly educational; Subscribed for more :)',
 'Great vid....SUBSCRIBED with the Bell👍',
 'Clicked for the learning, stayed for the hair,
 subscribed for the humor. Keep it up fam.',
 'I keep thinking the slinky on his head is going
 to fall off. But this stuff is too damn
 interesting so I subscribed anyways.']'

```

* tabular load a sql document to a dataframe.

```python
>>> pipe = Pipeline().append(docs.export('df'))

```
