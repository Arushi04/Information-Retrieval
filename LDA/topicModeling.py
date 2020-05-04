import argparse
import tqdm
import lda
from elasticsearch import Elasticsearch
import collections

import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from scipy.sparse import csr_matrix

def read_queries(fname):
    queries = {}
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split(",")
            qid, query = line[0], line[1:]
            queries[qid] = query
    print(queries.keys())
    return queries


def read_qrel(fname, ids):
    qrel = {}
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split()
            qid, docid= line[0], line[2]
            if qid in ids:
                if qid in qrel:
                    qrel[qid].add(docid)
                else:
                    qrel[qid] = {docid}
    return qrel


def read_bm25(fname, query):
    bm25_docs = set()
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split()
            qid, docid = line[0], line[2]
            if qid == query:
                if len(bm25_docs) == 1000:
                    return bm25_docs
                bm25_docs.add(docid)
    return bm25_docs

def divide_chunks(l, n=250):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_terms_from_ES(es, index, docs, stopwords):
    doc_terms = {}
    vocab = collections.Counter()
    results_chunk = divide_chunks(list(docs))
    for results in tqdm.tqdm(results_chunk):
        mterms = es.mtermvectors(index=index, doc_type='_doc', body=dict(ids=results,
                                          parameters=dict(term_statistics=True, fields=['text'])))['docs']

        for mterm in mterms:
            if "text" in mterm['term_vectors']:
                doc_id = mterm["_id"]
                for word in mterm['term_vectors']['text']['terms']:
                    if word in stopwords:
                        continue
                    tf = mterm['term_vectors']['text']['terms'][word]['term_freq']
                    if doc_id not in doc_terms:
                        doc_terms[doc_id] = collections.Counter()
                    doc_terms[doc_id][word] = tf
                    vocab[word] += tf
    '''
        doc_terms : {doc_id:{token:tf, token2:tf} ...}
        vocab : {token:freq}
    '''
    return doc_terms, vocab


def fit_model(doc_term_matrix, vocab, idx2doc):
    model = lda.LDA(n_topics=50, n_iter=500, random_state=1)
    model.fit(doc_term_matrix)
    topic_word = model.topic_word_
    n_top_words = 20
    topic_word_output = ''
    doc_topic_output = ''
    for i, topic_dist in enumerate(topic_word):
        print(i, topic_dist)
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        topic_word_output += ('Topic {}: {}\n'.format(i, ' '.join(topic_words)))

    doc_topic = model.doc_topic_
    print(doc_topic)
    for i in range(10):
        doc = idx2doc[i]
        print("{} (top topic: {})".format(doc, doc_topic[i].argmax()))
        doc_topic_output += ("{} (top topic: {}) \n ".format(doc, doc_topic[i].argmax()))

    return topic_word_output, doc_topic_output


def main(args):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    queries = read_queries(args.queries)  # Get the query numbers
    qrel = read_qrel(args.qrel, queries.keys()) #creates a qrel dict for all 25 queries
    f_top = open("all_topics_words.txt", "w")
    f_doc = open("all_doc_topics.txt", "w")
    for query in queries.keys():
        data = read_bm25(args.rfile, query)  # Read the BM25 and get top 1000 for each query
        data = data|qrel[query]  #union of BM25 and qrel docs for each query
        doc2idx, idx2doc, terms = {}, {}, {}
        doc_info = []
        stop_words = set(stopwords.words('english'))
        doc_terms, vocab = get_terms_from_ES(es, args.index, data, stop_words)
        for idx, i in enumerate(data):
            doc2idx[i] = idx
            idx2doc[idx] = i
        for idx, i in enumerate(vocab):
            terms[i] = idx

        for docid in doc_terms:
            for term in doc_terms[docid]:
                if term not in terms:
                    continue
                tf = doc_terms[docid][term]
                rid = doc2idx[docid]
                cid = terms[term]
                doc_info.append([rid, cid, tf])

        row, col, data = zip(*doc_info)
        doc_term_matrix = csr_matrix((data, (row, col)), shape=(len(doc_terms), len(vocab)))
        topic_words, doc_topics = fit_model(doc_term_matrix, tuple(vocab.keys()), idx2doc)

        f_top.write("Query : %s\n" %query)
        f_top.write(topic_words)
        f_top.write("-----------\n")

        f_doc.write("Query : %s\n" %query)
        f_doc.write(doc_topics)
        f_doc.write("-----------\n")
    f_top.close()
    f_doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index", type=str, default="ap_dataset1", help="")
    parser.add_argument("--rfile", type=str, default="data/okapi_bm25.out", help="") #rankedFile
    parser.add_argument("--qrel", type=str, default="data/qrel.txt", help="")
    parser.add_argument("--queries", type=str, default="data/queryfile.txt", help="")
    args = parser.parse_args()
    main(args)