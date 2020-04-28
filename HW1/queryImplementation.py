from __future__ import division
import os
import io
import math
from math import log
import argparse
from elasticsearch import Elasticsearch
import operator
import pickle

'''Setting default arguments and using these args to change the params through console if required
Default : python queryImplementation.py --queryfile queryfile1.txt --output default/
EC1 : python queryImplementation.py --queryfile queryfile_ec1.txt --output EC1/
EC2 : python queryImplementation.py --queryfile queryfile_ec2.txt --output EC2/
'''

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument("--index_name", type=str, default="hw1_dataset1", help="")
parser.add_argument("--max_search_size", type=int, default=85000, help="")
parser.add_argument("--k1", type=float, default=1.2, help="")
parser.add_argument("--k2", type=float, default=100, help="")
parser.add_argument("--b", type=float, default=0.75, help="")
parser.add_argument("--lmd", type=float, default=0.8, help="LAMBDA value")
parser.add_argument("--output", type=str, default="./default/", help="")
parser.add_argument("--stats_fname", type=str, default="precompute_Stats.nb", help="")
parser.add_argument("--queryfile", type=str, default="queryfile1.txt", help="")
args = parser.parse_args()
k1, k2 = args.k1, args.k2
LAMBDA, b = args.lmd, args.b
global_stats = pickle.load(open(args.stats_fname,"rb"))

if not os.path.exists(args.output):
    os.makedirs(args.output)

#Setting up the ES connection
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
dir_path = "/Users/arushi/PycharmProjects/InformationRetrieval/HW1"
filename = args.queryfile

#Getting the vocab size
query_body = {"aggs": {"vocabSize": {"cardinality": {"field": "text"}}}, "size": 0}
search_result = es.search(index=args.index_name, body=query_body)
V = search_result['aggregations']['vocabSize']['value']
print("Vocab size : %s" % V)


def write_to_file(qid, result_list, fname):
    # sort result list by score
    fname = os.path.join(args.output, fname)
    print("\t\t\twriting query %s to %s" % (qid, fname))
    result_list = sorted(result_list, key=operator.itemgetter(1), reverse=True)
    with open(fname, "a") as fout:
        for idx, res in enumerate(result_list[:1000]):
            # print(qid, "Q0", res[0], idx+1, res[1], "ES" )
            final_input = str(qid) + " Q0 " + str(res[0]) + " " + str(idx + 1) + " " + str(res[1]) + " Exp \n"
            fout.write(final_input)


def modify_query(query):
    qresult = es.indices.analyze(index=args.index_name, body={'analyzer': 'stopped', 'text': query})
    query_terms = [t['token'] for t in qresult['tokens']]
    return " ".join(query_terms)


def run_es_builtin(query):
    print("\t\tRunning built-in")
    rank_id_list = []
    query_body = {"query": {"bool": {"must": {"match": {"text": query}}}}}

    result = es.search(index=args.index_name, body=query_body, size=args.max_search_size)
    data = [doc for doc in result['hits']['hits']]
    for doc in data:
        doc_id = doc['_id']
        score = doc['_score']
        rank_id_list.append([doc_id, score])
    return rank_id_list


def run_all_models(query, doc_ids):
    print("\t\tRunning models")
    okapi_tf_list, tf_idf_list, okapi_bm25_list, laplace_list, lm_jm_list = [], [], [], [], []

    query_terms = query.split()

    for doc_id in doc_ids:
        okapi_tf, tf_idf, okapi_bm25, laplace, lm_jm = 0.0, 0.0, 0.0, 0.0, 0.0

        total_doc_len = global_stats['sum_ttf']
        D = global_stats['doc_count']
        avg_doc_len = (total_doc_len / D)  # 248.65

        doc_len = 0

        for term in global_stats['term_freq'][doc_id]:
            doc_len += global_stats['term_freq'][doc_id][term]

        for query_term in query_terms:
            if query_term in global_stats['term_freq'][doc_id]:
                tf = global_stats['term_freq'][doc_id][query_term]
                df = global_stats['doc_freq'][query_term]
                tfd = global_stats['ttf'][query_term]

                # Calculating Okapi-tf
                okapi_base = tf / (tf + 0.5 + (1.5 * (doc_len / avg_doc_len)))
                okapi_tf += okapi_base

                # Calculating tf-idf
                tf_idf += (okapi_base * (log(D / df)))

                # Calculating Okapi BM25
                tfq = query_terms.count(query_term)
                first_term = (log((D + 0.5) / (df + 0.5)))
                second_term = ((tf + (k1 * tf)) / (tf + k1 * ((1 - b) + (b * (doc_len / avg_doc_len)))))
                third_term = ((tfq + (k2 * tfq)) / (tfq + k2))
                okapi_bm25 += (first_term * second_term * third_term)

                # Calculating Laplace Smoothing
                p_laplace = (tf + 1) / (doc_len + V)
                laplace += log(p_laplace)

                # Calculating JM Smoothing
                p_jm = (LAMBDA * (tf / doc_len)) + ((1 - LAMBDA) * (tfd / total_doc_len))
                lm_jm += log(p_jm)

            else:
                tf = 0.0
                # Calculating Laplace Smoothing
                p_laplace = 1 / (doc_len + V)
                laplace += log(p_laplace)

                # Calculating JM Smoothing

                tfd = global_stats['ttf'][query_term] if query_term in global_stats['ttf'] else 1
                p_jm = (1 - LAMBDA) * (tfd / total_doc_len)
                lm_jm += log(p_jm)

        okapi_tf_list.append([doc_id, okapi_tf])
        tf_idf_list.append([doc_id, tf_idf])
        okapi_bm25_list.append([doc_id, okapi_bm25])
        laplace_list.append([doc_id, laplace])
        lm_jm_list.append([doc_id, lm_jm])

    assert len(okapi_tf_list) == len(doc_ids)
    return [okapi_tf_list, tf_idf_list, okapi_bm25_list, laplace_list, lm_jm_list]


query_id = ""
query = []
query_dict = {}
filepath = os.path.join(dir_path, filename)

# Parsing and storing the query no and main query in dict
with io.open(filepath, 'r', encoding='ISO-8859-1') as f:
    for line in f:
        line = line.strip()
        line = line.split(",")
        query_id = line[0]
        query_text = ",".join(line[1:])
        query_dict[query_id] = query_text

for qid in query_dict:
    current_query = query_dict[qid]
    print("\n\nQuery : %s : %s" % (qid, current_query))
    current_query = modify_query(current_query)
    print("\tModified Query: %s" % current_query)

    result_list = run_es_builtin(current_query)  # function call for ES Built in
    write_to_file(qid, result_list, "es-built-in.out")

    result_list = run_all_models(current_query, [ids[0] for ids in result_list])
    write_to_file(qid, result_list[0], "okapi-tf.out")
    write_to_file(qid, result_list[1], "tf-idf.out")
    write_to_file(qid, result_list[2], "okapi_bm25.out")
    write_to_file(qid, result_list[3], "laplace.out")
    write_to_file(qid, result_list[4], "lm-jm.out")