# metrics 📊

**METRICS-DOCS** the following is a description of each method from `david.pipeline._.metric.TextMetrics` class.

call the `get_all_metrics` instance method to quickly get basic stats on the texts.

```markdown
* String-level    -> if `string=True`:

  - stringLength        : sum of all words in a string.

* Word-level      -> if `words=True`:
  
  - avgWordLength       : average number of words.
  - isStopwordCount     : count of stopwords only.
  - noStopwordCount     : count of none stopwords.

* Character-level -> if `character=True`:

  - charDigitCount      : count of digits chars.
  - charUpperCount      : count of uppercase chars.
  - charLowerCount      : count of lowercase chars.

* Sentiment-level -> if `sentiment=True`:

  - sentiPolarity       : polarity score with Textblob, (float).
  - sentiSubjectivity   : subjectivity score with Textblob (float).
  - sentimentLabel      : labels row with one (pos, neg, neutral) tag.

* Tag-extraction  -> if `tags=True`:

  - authorTimeTag       : extracts video time tags, e.g. 1:20.
  - authorUrlLink       : extracts urls links if found.
  - authorHashTag       : extracts hash tags, e.g. #numberOne.
  - authorEmoji         : extracts emojis  👾.
```

```python
pipe.get_all_metrics(string=True, words=True, characters=True, tags=True)
pipe.describe()
```

```ipython
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

* with `tags=True` the following attributes are available. (the amout of tags varies on the size of the dataset)

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
