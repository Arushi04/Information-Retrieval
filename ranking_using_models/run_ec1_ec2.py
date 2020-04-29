from __future__ import division
import argparse
from elasticsearch import Elasticsearch
from math import log
import pickle
import operator
from collections import Counter

'''Setting default arguments and using these args to change the params through console if required
EC1 : python run_ec1_ec2.py --outf queryfile_ec1.txt --method ec1 --cutoff_per_query 3
EC2 : python run_ec1_ec2.py --outf queryfile_ec2.txt --method ec2 --cutoff_per_query 1 --query_fname queryfile_ec1.txt
'''

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument("--index_name", type=str, default="hw1_dataset1", help="")
parser.add_argument("--stats_fname", type=str, default="precompute_Stats.nb", help="")
parser.add_argument("--query_fname", type=str, default="queryfile1.txt", help="")
parser.add_argument("--model_fname", type=str, default="default/tf-idf.out", help="")
parser.add_argument("--top_k", type=int, default=10, help="")
parser.add_argument("--cutoff_per_query", type=int, default=3, help="")
parser.add_argument("--method", type=str, default="ec1", help="")
parser.add_argument("--outf", type=str, default="queryfile_ec1.txt", help="")

args = parser.parse_args()

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
global_stats = pickle.load(open(args.stats_fname, "rb"))

# Reads the top k documents of the tf-idf output file. Keys: queryid, values: docids
def get_docids(topk):
    output = {}
    with open(args.model_fname) as fin:
        for line in fin:
            line = line.strip().split()
            if line[0] not in output:
                output[line[0]] = [line[2]]
            else:
                output[line[0]].append(line[2])

    for key in output:
        output[key] = output[key][:topk]
    return output


# finds all the common terms across top k docs. search results :list of docs
def all_common(search_results, query_terms):
    common = None
    for search_docid in search_results:
        terms = global_stats['term_freq'][search_docid]
        if common is None:
            common = terms
        else:
            common = list(set(terms) & set(common))
    common = [i for i in common if i not in query_terms]
    print("\t\tCommon Terms Found: %d" % len(common))
    return common[:args.cutoff_per_query]


def top_common(search_results, query_terms):
    common = Counter()
    for search_docid in search_results:
        terms = global_stats['term_freq'][search_docid]
        for term in terms:
            common[term] += 1

    terms = []
    for key, value in sorted(common.items(), key=lambda item: item[1], reverse=True)[:args.cutoff_per_query + 10]:
        print("\t\tTop Common : %s : %s" % (key, value))
        terms.append(key)

    terms = [i for i in terms if i not in query_terms]
    return terms[:args.cutoff_per_query]


def get_SignificantTerms(query_term):
    query_body = {
        "query": {
            "match": {"text": query_term}
        },
        "aggregations": {
            "significant_crime_types": {
                "significant_terms": {
                    "field": "text"
                }
            }
        },
    }

    result = es.search(index=args.index_name, body=query_body)
    terms = [[i['key'], i['score']] for i in result['aggregations']['significant_crime_types']['buckets']]
    return terms


def run_ec2(terms):
    D = global_stats["doc_count"]
    new_terms = []
    for term in terms:
        synonyms = get_SignificantTerms(term)
        for sym, score in synonyms:
            if not sym[0].isalpha():
                continue
            idf = log(D / global_stats["doc_freq"][term])
            new_terms.append([term, sym, idf, score])

    new_terms = sorted(new_terms, key=lambda item: (item[2], item[3]), reverse=True)
    new_terms = [a for a in new_terms if a[1] not in terms]
    new_terms_unique = []
    for term, sym, score, idf in new_terms:
        if sym not in [i[1] for i in new_terms_unique]:
            new_terms_unique.append([term, sym, score, idf])
    new_terms = new_terms_unique
    for term, sym, score, idf in new_terms[:10]:
        print("\t\t\t (Term:Syn:idf:score) --- (%s:%s:%f:%f)" % (term, sym, score, idf))
    new_terms = [i[1] for i in new_terms]

    return new_terms[:args.cutoff_per_query]


def get_query_terms(text):
    qresult = es.indices.analyze(index=args.index_name, body={'analyzer': 'stopped', 'text': text})
    query_terms = [t['token'] for t in qresult['tokens']]
    return query_terms


fout = open(args.outf, "w")
ranked_docs = get_docids(args.top_k)
with open(args.query_fname) as f:
    for line in f:
        line = line.strip().split(",")
        qid, query = line[0], ",".join(line[1:])
        query_terms = get_query_terms(query)
        print("Query: %s" % qid)
        print("Original Query: %s" % query)
        print("Query Terms: %s" % " ".join(query_terms))
        search_results = ranked_docs[qid]


        if args.method == "ec1":
            new_terms = all_common(search_results, query_terms)
            new_terms = top_common(search_results, query_terms) + new_terms
            new_terms = list(set(new_terms))
        elif args.method == "ec2":
            new_terms = run_ec2(query_terms)
        else:
            raise Exception("method type wrong")
        print("\t\t%d New Terms: %s" % (len(new_terms), " ".join(new_terms)))
        print("\t\tECx Query: %s %s" % (query, " ".join(new_terms)))
        fout.write("%s,%s %s\n" % (qid, query, " ".join(new_terms)))

fout.close()
