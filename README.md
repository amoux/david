# david - Vuepoint Analytics

* David is an NLP toolkit implemented with Gensim, Tensorflow, PyTorch, NLTK, and spaCy among other open-source libraries. At Vuepoint, we are applying models built on these libraries for text-classification, intent-detection, topic-modeling to improve youtube content creators view of their audience to produce more suitable content.

![image](https://fromdirectorstevenspielberg.com/wp-content/uploads/2017/07/15.jpg?raw=true)

*The goal for David is to assist content creators to increase the exposure on YouTube and their videos in the presence of more viewers. From live/historical textual information.*

> Currently the project is still in beta stages. Once the library is stable enough, an enhanced realease will be available for the public to use.

* configure the database and build a dataset from a search query. default parameters `db_name='comments_v2.db', table='comments'`.

```python
from david_server.sql import SqliteCommentsDB
sql = SqliteCommentsDB()
docs = sql.get_similartexts('i subscribed')
[doc.text for doc in docs][:5]
```

```
['Video was hilarious, subscribed!',
 'Hamazingly educational; Subscribed for more :)',
 'Great vid....SUBSCRIBED with the Bell👍',
 'Clicked for the learning, stayed for the hair,
 subscribed for the humor. Keep it up fam.',
 'I keep thinking the slinky on his head is going
 to fall off. But this stuff is too damn
 interesting so I subscribed anyways.']
 ```

* export a document to a dataframe with the `export` attribute.

```python
from david.pipeline import Pipeline
pipe = Pipeline(docs.export('df'))
```

* call the `get_all_metrics` instance method to quickly get basic stats on the texts. for more information on the additional parameters available [click here]().

```python
pipe.get_all_metrics(string=True, words=True, characters=True, tags=True)
pipe.describe()
```

```
       stringLength  avgWordLength  ...  charUpperCount  charLowerCount
count     88.000000      88.000000  ...       88.000000       88.000000
mean     156.818182       5.916454  ...        6.375000      116.704545
std      162.526924       1.176320  ...        7.886133      124.026986
min       12.000000       4.166667  ...        0.000000       10.000000
25%       49.000000       5.164435  ...        2.000000       37.000000
50%      103.000000       5.557041  ...        3.000000       70.500000
75%      199.500000       6.375000  ...        8.000000      147.250000
max      836.000000       9.000000  ...       46.000000      647.000000

[8 rows x 7 columns]
```

* with tags=True the following attributes are available. (the amout of tags varies on the size of the dataset)

```python
pipe.authorEmoji.unique()
```

```
array(['👍', '😍😍', '😂💙👄', '😊', '💕💕💕', '✌🏾', '😙', '🤔🤷♂'],
      dtype=object)
```

```python
pipe.authorUrlLink.unique()
```

```
array([nan, 'https://www.youtube.com/channel/UCywXyzx6GZpDyxRvMOqLMiw'],
      dtype=object)
```

```python
pipe.authorHashTags.unique()
```

```
array([nan, '#SUBSCRIBED'], dtype=object)
```

```python
pipe.authorTimeTag.unique()
```

```
array([nan, '10:06'], dtype=object)
```
