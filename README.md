# david nlp  💬 (social-media-toolkit)

The goal of this toolkit is to speed the time-consuming steps to obtain, store, and pre-process textual data from social-media-sites like youtube, and implementing natural language processing techniques to extract highly specific information, such as the indications of product or service trends and similar.

- Projects using this toolkit:

  - **vuepoint** (flask-web-app)
    - Vuepoint is an analytics web-application that will help content creators automate the tedious task of manually having to scroll and read through multiple pages of comments to understand what an audience wants or dislikes. Extracting the vital information that is relevant to each content creator quickly.

  - **david-sentiment** (embedding-models):
    - Train an accurate sentiment model based on `meta-learning` and `embedding` techniques with small datasets and a few lines of code.

  - **qaam-nlp** (question-answering): 
    - Given an article or blog URL with text content. The model (based on `BERT`) will answer questions on the given context or subject.

- 📃 Objectives:
  - Build a rich toolkit for **end-to-end** ***SM*** pipelines.
  - Specialized text preprocessing techniques for social-media text.
  - SM Data: Currently working only on ***YouTube***, but then plan to extend to similar sites like: *Reddit*, *Twitter* in the future.

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
* vocab_index: [('.', 1), ('the', 2), (',', 3), ('to', 4), ('i', 5)]
* vocab_count: [('.', 2215), ('the', 2102), (',', 1613), ('to', 1286), ('i', 1277)]
```

- Converting ***this-to-that*** and ***that-to-this*** `(encoding|decoding)` ♻

```python
text = "Hello, this is a text! embedded with youtube comments 😊"
str2ids = tokenizer.convert_string_to_ids(text)
ids2tok = tokenizer.convert_ids_to_tokens(str2ids)
tok2str = tokenizer.convert_tokens_to_string(ids2tok)
assert len(text) == len(tok2str)
```

- If a token is missing then it's because is missing from the vocabulary. To add a new token, pass a single token to `Tokenizer.add_token()` instance method (see below).

```python
* str2ids: [659, 3, 17, 9, 7, 1446, 18, 2648, 21, 391, 766, 787]
* ids2tok: ['hello', ',', 'this', 'is', 'a', 'text', '!', 'embedded', 'with', 'youtube', 'comments', '😊']
* tok2str: "hello, this is a text! embedded with youtube comments 😊"
```

- Add a new token.

```python
emoji_tok = ["👻"]
emoji_tok[0] in tokenizer.vocab_index
...
False
# add the missing token
tokenizer.add_token(emoji_tok)
# token has been added and indexed
tokenizer.convert_tokens_to_ids(emoji_tok)
...
[7674]
```

- Embedding an iterable document of string sequences.

```python
sequences = tokenizer.document_to_sequences(document=train_data)  # returns a generator
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

Proper documentation for the `david.text` module will be added as soon. In the meantime, here are the main text preprocessing functions I use every day (daily actually!) for my NLP tasks. I frequently look for approaches to optimize and enhance these methods to the smallest detail!

```bash
## Text preprocessing methods
  clean_tokenization
  get_sentiment_polarity
  get_sentiment_subjectivity
  nltk_word_tokenizer
  normalize_whitespace
  normalize_wiggles
  part_of_speech_annotator
  part_of_speech_lemmatizer
  preprocess_doc
  preprocess_sequence
  remove_punctuation
  remove_repeating_characters
  remove_repeating_words
  remove_stopwords
  remove_urls
  spacy_token_lemmatizer
  string_printable
  treebank_to_wordnet_pos
  unicode_to_ascii

## Summarizer built with spaCy
  summarizer

## Other/Text utility methods
  build_wordcloud
  clean_tokens
  complete_sentences
  extract_text_from_url
  get_vocab_size
  is_tokenized_doc
  largest_sequence
  split_train_test
```

## david.youtube

Main YouTube scraper - No `API` needed! It scrapes comments from a `video_id` or a `video_url`. If you want to extract before passing. the utility function `david.youtube.utils.extract_videoid(url: str)` can extract the `video_id` (including broken urls).

```python
from david.youtube import YTCommentScraper
scraper = YTCommentScraper()
scrape_job = scraper.scrape_comments(video_id="bCtOFZoCvBE", video_url=None)  # returns a generator
comments = list(scrape_job)  # initialize the scraper!
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
