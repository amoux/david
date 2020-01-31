import spacy
from david import CommentsSql, SimilarDocuments

MIN_TEXT_LEN = 30
MAX_TEXT_LEN = 200
NGRAM_RANGE = (1, 3)
TOP_K = 10
DB_QUERY = "%make a video%"
SD_QUERY = "Make a video on the new iphone"


def prep_text_batch(batch):
    """ Preprocess the texts fetched from the database """
    texts = []
    for q in batch:
        q = q.text.strip()
        if len(q) > MIN_TEXT_LEN and len(q) < MAX_TEXT_LEN and q not in texts:
            texts.append(q)
    return texts


def to_sentences(texts):
    """ Transform an interable of texts to an iterable of sentences """
    nlp = spacy.load('en_core_web_sm')
    sentences = []
    for doc in nlp.pipe(texts):
        for i in doc.sents:
            if i.text and i.text not in sentences:
                sentences.append(i.text)
    return sentences


def main():
    db = CommentsSql('unbox')
    batch = db.fetch_comments(DB_QUERY, 'id,text', 'id')
    # preprocess and filter all the comments fetched from the database.
    texts = prep_text_batch(batch)
    sents = to_sentences(texts)
    # construct a SimilarDocuments object with the sentences and ngram range.
    sd = SimilarDocuments(sents, ngram=NGRAM_RANGE)
    sd.learn_vocab()
    # print out the top n most similar documents to the query.
    for doc in sd.iter_similar(TOP_K, SD_QUERY, True):
        if doc['sim']:
            print("sim: {} text: {}".format(doc['sim'], doc['text']))

    print(f"\nsimilar to query: {sd.queries}\n")


if __name__ == '__main__':
    main()
