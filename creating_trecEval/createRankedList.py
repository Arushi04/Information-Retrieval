from __future__ import division
import os
import argparse
from elasticsearch import Elasticsearch
import operator


#Setting up the ES connection
#es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch("https://96aa4157ead74b5ca4926523b1d1994e.us-east-1.aws.found.io:9243", http_auth=('elastic', 'MrkfJ5hxIcCOzTMfOa1Nftzy'))


def write_to_file(qid, result_list, fname):
    # sort result list by score
    fname = os.path.join(args.output, fname)
    print("\t\t\twriting query %s to %s" % (qid, fname))
    result_list = sorted(result_list, key=operator.itemgetter(1), reverse=True)
    with open(fname, "a") as fout:
        for idx, res in enumerate(result_list[:200]):
            # print(qid, "Q0", res[0], idx+1, res[1], "ES" )
            final_input = str(qid) + " Q0 " + str(res[0]) + " " + str(idx + 1) + " " + str(res[1]) + " ES \n"
            fout.write(final_input)


def modify_query(query):
    qresult = es.indices.analyze(index=args.index_name, body={'analyzer': 'stopped', 'text': query})
    query_terms = [t['token'] for t in qresult['tokens']]
    return " ".join(query_terms)


def run_es_builtin(query, index, max_search_size):
    print("\t\tRunning built-in")
    rank_id_list = []
    query_body = {"query": {"bool": {"must": {"match": {"text": query}}}}}

    result = es.search(index=index, body=query_body, size=max_search_size, request_timeout=10)
    data = [doc for doc in result['hits']['hits']]
    for doc in data:
        doc_id = doc['_id']
        score = doc['_score']
        rank_id_list.append([doc_id, score])
    return rank_id_list


def main(args):
    query_dict = {'151901': 'College of Cardinals', '151902':'Ten Commandments', '151903':'recent popes'}

    for qid in query_dict:
        current_query = query_dict[qid]
        print("\n\nQuery : %s : %s" % (qid, current_query))
        current_query = modify_query(current_query)
        print("\tModified Query: %s" % current_query)

        result_list = run_es_builtin(current_query, args.index_name, args.max_search_size)  # function call for ES Built in
        write_to_file(qid, result_list, args.fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index_name", type=str, default="ir_hw03_prod2", help="")
    parser.add_argument("--max_search_size", type=int, default=1000, help="")
    parser.add_argument("--output", type=str, default="output/", help="")
    parser.add_argument("--fname", type=str, default="esbuiltin.txt", help="")
    args = parser.parse_args()
    main(args)