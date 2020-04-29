import sys
from elasticsearch import Elasticsearch
import pickle


def divide_chunks(l, n=250):
    for i in range(0, len(l), n):
        yield l[i:i + n]


term_freq = {}
doc_freq = {}
ttf = {}
doc_count = None
sum_ttf = None

INDEX_NAME = "ap_dataset"
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
empty_body = {
    'query':
        {
            "match_all": {}
        }
}

results = es.search(index=INDEX_NAME, body=empty_body, size=85000)['hits']['hits']
results = [result['_id'] for result in results]
print("size:%d" % len(results))

results_chunk = divide_chunks(results)
for results in results_chunk:
    mterms = es.mtermvectors(index=INDEX_NAME, doc_type='_doc',
                             body=dict(ids=results,
                                parameters=dict(term_statistics=True, field_statistics=True, fields=['text'])))['docs']

    for mterm in mterms:
        if "text" in mterm['term_vectors']:  # for handling empty docs
            for term in mterm['term_vectors']['text']['terms']:
                if doc_count is None:
                    doc_count = mterm['term_vectors']['text']['field_statistics']['doc_count']
                if sum_ttf is None:
                    sum_ttf = mterm['term_vectors']['text']['field_statistics']['sum_ttf']

                if mterm["_id"] not in term_freq:
                    term_freq[mterm["_id"]] = {}
                if term not in term_freq[mterm["_id"]]:
                    term_freq[mterm["_id"]][term] = {}

                term_freq[mterm["_id"]][term] = mterm['term_vectors']['text']['terms'][term]['term_freq']
                doc_freq[term] = mterm['term_vectors']['text']['terms'][term]['doc_freq']
                ttf[term] = mterm['term_vectors']['text']['terms'][term]['ttf']

        else:
            print(mterm['_id'])

# Saving it in format -  term_freq = {doc_id1:{term1:2, term2:3,....}, doc_id2:{term...}}

data = {'term_freq': term_freq, 'doc_freq': doc_freq, 'ttf': ttf, 'sum_ttf': sum_ttf, 'doc_count': doc_count}
pickle.dump(data, open("precompute_stats.nb", "wb"))
