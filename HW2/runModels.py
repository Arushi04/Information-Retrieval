from __future__ import division
import os
import io
from math import log, exp, inf
import argparse
import operator
import pickle
import tqdm
import subprocess
import pandas as pd
import numpy as np

from custom_util import my_tokenize, read_stopwords, get_stats

'''Setting default arguments and using these args to change the params through console if required
Default : python runModels.py --queryfile data/queryfile1.txt --output output/models/stemmed/
python runModels.py \
-qf data/queryfile1.txt \
-stem \
--compress \
-i output/stemmed_compressed/combined_inverted_index_84.txt \
-vocab output/stemmed_compressed/vocab.pickle \
-c output/stemmed_compressed/combined_catalog_84.txt \
--ptype min \
--p_alpha .1 \
-o output/models/stemmed_compressed/ 

Unstemmed : python runModels.py --queryfile data/queryfile1.txt --vocab_fname output/unstemmed/vocab.pickle --catalog output/unstemmed/combined_catalog_84.txt -o output/models/unstemmed/
'''

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument("-i", "--inverted_index", required=True, type=str,
                    default="output/stemmed/combined_inverted_index_84.txt", help="")
parser.add_argument("-c", "--catalog", type=str, default="output/stemmed/combined_catalog_84.txt", help="")
parser.add_argument("--ptype", type=str, default="min", help="Proximity Type : min/max/avg")
parser.add_argument("--p_alpha", type=float, default=0.3, help="Alpha for Proximity Retrieval Equation")
parser.add_argument("--k1", type=float, default=1.2, help="")
parser.add_argument("--k2", type=float, default=100, help="")
parser.add_argument("--b", type=float, default=0.75, help="")
parser.add_argument("--lmd", type=float, default=0.8, help="LAMBDA value")
parser.add_argument("-vocab", "--vocab_fname", type=str, default="output/stemmed/vocab.pickle", help="")
parser.add_argument("-o", "--output", type=str, default="output/models1/", help="")
parser.add_argument("-qf", "--queryfile", type=str, default="data/queryfile.txt", help="")
parser.add_argument("-stem", "--do_stem", action='store_true', help="")
parser.add_argument("--stopfile", type=str, default="data/stoplist.txt", help="")
parser.add_argument("--compress", action='store_true', help="")

args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)
else:
    # remove all log files
    import glob, os

    for f in glob.glob(args.output + "/" + "*.out"):
        print("Deleting : %s" % f)
        os.remove(f)

k1, k2 = args.k1, args.k2
LAMBDA, b = args.lmd, args.b

# Reading Vocab and Stopwords
vocab = pickle.load(open(args.vocab_fname, "rb"))
SUM_TTF, DOC_COUNT, DOC_LEN_MAP = vocab["sum_ttf"], vocab["doc_count"], vocab["doc_len"]
doc_vocab, vocab = vocab["doc_vocab"], vocab["token_vocab"]
stopwords = read_stopwords(args.stopfile)

# Reading Catalog and Inverted_Index
catalog = pickle.load(open(args.catalog, "rb"))
inv_index_fin = open(args.inverted_index, "rb")

# Getting the vocab size
V = len(vocab)
print("Vocab size : %s" % V)
cache = {}


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


def get_postings(term, docid=None):
    if term in cache:
        postings = cache[term]
    else:
        postings = get_stats(catalog[term], inv_index_fin, term, args.compress)
        cache[term] = postings

    if docid is None:
        return postings
    if docid in postings:
        return postings[docid]
    else:
        return None


def compute_proximity(termA, termB, docID):
    postA = get_postings(termA, docID)
    postB = get_postings(termB, docID)

    dist = None

    if postA is None or postB is None:
        return dist
    assert len(postA) > 0
    assert len(postB) > 0

    for i in postA:
        for j in postB:
            diff = abs(i - j) + 1
            if dist is None:
                dist = diff
            else:
                diff = min(dist, diff)
    return diff


def get_proximity_distance(qterms, docid, ptype):
    pscores = []
    qterms = list(set(qterms))
    for idx in range(len(qterms) - 1):
        first_term = qterms[idx]
        for jdx in range(idx + 1, len(qterms)):
            second_term = qterms[jdx]
            current_dist = compute_proximity(first_term, second_term, docid)
            if current_dist is not None:
                pscores.append(current_dist)

    if len(pscores) == 0:
        return None
    if ptype == "min":
        return min(pscores)
    elif ptype == "max":
        return max(pscores)
    elif ptype == "avg":
        return sum(pscores) / len(pscores)
    else:
        raise Exception("proximity aggregation type is incorrect.")


def get_precision(command):
    output = subprocess.check_output(command.split()).decode("utf-8")
    flag = False
    for line in output.split("\n"):
        line = line.strip()
        if flag:
            precision = float(line)
            flag = False
        if line.startswith("Average precision (non-interpolated)"):
            flag = True
    return precision


def modify_query(query, do_stem, stopwords):
    query_terms = my_tokenize(query, do_stem, stopwords)
    return " ".join(query_terms)


def run_all_models(query):
    print("\t\tRunning models")
    okapi_tf_list, tf_idf_list, okapi_bm25_list, laplace_list, lm_jm_list = [], [], [], [], []
    proximity_list = []

    query_terms = query.split()

    for doc_name in tqdm.tqdm(doc_vocab["ntoi"]):
        doc_id = doc_vocab["ntoi"][doc_name]
        okapi_tf, tf_idf, okapi_bm25, laplace, lm_jm, pro_bm25 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        total_doc_len = SUM_TTF
        D = DOC_COUNT
        avg_doc_len = (total_doc_len / D)

        doc_len = DOC_LEN_MAP[doc_name]
        if doc_len == 0:
            continue
        proximity_dist = get_proximity_distance(query_terms, doc_id, args.ptype)
        if proximity_dist == None:
            proximity_dist = total_doc_len

        proximity_ret_score = log(args.p_alpha + exp(-1 * proximity_dist))

        for query_term in query_terms:

            tf, df, tfd = 0, 0, 1.0
            if query_term in catalog:
                postings = get_postings(query_term)

                if postings is not None and doc_id in postings:
                    # print(query_term, doc_id, postings)
                    tf = len(postings[doc_id])
                    df = len(postings)
                    tfd = postings["ttf"]
            # tf should be 0.0 if not found in that document, tfd should be 1 (for smoothing purpose)
            if query_term in catalog and tf:
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

                # Calculating Proximity Okapi
                pro_bm25 += (first_term * second_term * third_term)

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
                p_jm = (1 - LAMBDA) * (tfd / total_doc_len)
                lm_jm += log(p_jm)

        okapi_tf_list.append([doc_name, okapi_tf])
        tf_idf_list.append([doc_name, tf_idf])
        okapi_bm25_list.append([doc_name, okapi_bm25])
        laplace_list.append([doc_name, laplace])
        lm_jm_list.append([doc_name, lm_jm])
        proximity_list.append([doc_name, tf_idf + proximity_ret_score])

    return [okapi_tf_list, tf_idf_list,
            okapi_bm25_list, laplace_list,
            lm_jm_list, proximity_list]


query_id = ""
query = []
query_dict = {}

# Parsing and storing the query no and main query in dict
with io.open(args.queryfile, 'r', encoding='ISO-8859-1') as f:
    for line in f:
        line = line.strip()
        line = line.split(",")
        query_id = line[0]
        query_text = ",".join(line[1:])
        query_dict[query_id] = query_text

for qidx, qid in enumerate(query_dict):
    current_query = query_dict[qid]
    print("\n\n[%d/%d]Query : %s : %s" % (qidx + 1, len(query_dict), qid, current_query))
    current_query = modify_query(current_query, args.do_stem, stopwords)
    print("\tModified Query: %s" % current_query)
    result_list = run_all_models(current_query)

    write_to_file(qid, result_list[0], "okapi-tf.out")
    write_to_file(qid, result_list[1], "tf-idf.out")
    write_to_file(qid, result_list[2], "okapi_bm25.out")
    write_to_file(qid, result_list[3], "laplace.out")
    write_to_file(qid, result_list[4], "lm-jm.out")
    write_to_file(qid, result_list[5], "proximity.out")
