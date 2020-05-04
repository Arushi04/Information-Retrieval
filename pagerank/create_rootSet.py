from __future__ import division
import os
import argparse
from elasticsearch import Elasticsearch
import operator


'''Setting default arguments and using these args to change the params through console if required
Default : python queryImplementation.py --index_name hw1_dataset --queryfile queryfile1.txt --output default/
EC1 : python queryImplementation.py --queryfile queryfile_ec1.txt --output EC1/
EC2 : python queryImplementation.py --queryfile queryfile_ec2.txt --output EC2/
'''

#Setting up the ES connection
#es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
es = Elasticsearch("https://96aa4157ead74b5ca4926523b1d1994e.us-east-1.aws.found.io:9243", http_auth=('elastic', 'MrkfJ5hxIcCOzTMfOa1Nftzy'))


def write_to_file(result_list, fname):
    # sort result list by score
    fname = os.path.join(args.output, fname)
    print("\t\t\twriting query to %s" % (fname))
    result_list = sorted(result_list, key=operator.itemgetter(1), reverse=True)
    with open(fname, "a") as fout:
        for idx, res in enumerate(result_list):
            print("res : ", res[0], res[1])
            final_input = str(str(res[0]) + " " + str(res[1]) + " \n")
            fout.write(final_input)


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
    query = '1519 catholic church'

    result_list = run_es_builtin(query, args.index_name, args.max_search_size)  # function call for ES Built in
    write_to_file(result_list, args.fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index_name", type=str, default="ir_hw03_prod2", help="")
    parser.add_argument("--max_search_size", type=int, default=3000, help="")
    parser.add_argument("--output", type=str, default="output/", help="")
    parser.add_argument("--fname", type=str, default="esbuiltin.txt", help="")
    args = parser.parse_args()
    main(args)