import argparse
import pickle
import tqdm
import collections

from elasticsearch import Elasticsearch

def divide_chunks(l, n=250):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_unigrams_from_ES(es, index, split_type):
    query_body = {"query": {"bool": {"must": {"match": {"split": split_type}}}}}
    results = es.search(index=index, body=query_body, size=85000)['hits']['hits']
    results = [result['_id'] for result in results]
    print("size of retrieved result :%d" % len(results))
    data = collections.Counter()
    vocab = collections.Counter()
    results_chunk = divide_chunks(results)
    for results in tqdm.tqdm(results_chunk):
        mterms = es.mtermvectors(index=index, doc_type='_doc',
                                 body=dict(ids=results,
                                           parameters=dict(term_statistics=True,
                                                           fields=['text'])))['docs']

        for mterm in mterms:
            if "text" in mterm['term_vectors']:
                doc_id = mterm["_id"]
                for word in mterm['term_vectors']['text']['terms']:
                    tf = mterm['term_vectors']['text']['terms'][word]['term_freq']
                    if doc_id not in data:
                        data[doc_id] = collections.Counter()
                    data[doc_id][word] = tf
                    vocab[word] += tf

    '''
        train : {doc_id:{token:tf, token2:tf} ...}
        test : {doc_id:{token:tf, token2:tf} ...}
        train vocab : {token:freq}
    '''
    return data, vocab


def main(args):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    train_data, vocab = get_unigrams_from_ES(es, args.index, "train")
    test_data, _ = get_unigrams_from_ES(es, args.index, "test")
    with open(args.savepath+"unigram.train.stats", 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.savepath+"unigram.test.stats", 'wb') as handle:
        pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    vocab = sorted(vocab.items(), key=lambda pair: pair[1], reverse=True)
    with open("%s/train.vocab.txt"%args.savepath, "w") as handle:
        for w in vocab:
            handle.write("%s %d\n"%(w[0], w[1]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index", type=str, default="spam_data", help="")
    parser.add_argument("--savepath", type=str, default="./data/", help="")
    args = parser.parse_args()
    main(args)