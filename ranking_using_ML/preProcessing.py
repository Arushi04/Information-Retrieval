import argparse
import random
import os
import operator

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression


random.seed(4)

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
            qid, docid, rel = line[0], line[2], int(line[3])
            if qid in ids:
                if qid in qrel:
                    qrel[qid][docid] = [rel]
                else:
                    qrel[qid] = {docid : [rel]}
    return qrel


def read_additional(fname, qrel, k):
    non_relevant_docs = {}
    for qid in qrel:
        non_relevant_docs[qid] = len([doc for doc in qrel[qid] if qrel[qid][doc][0]==0])

    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split()
            qid, docid = line[0], line[2]
            if qid not in qrel:
                continue
            if docid in qrel[qid]:
                continue

            if non_relevant_docs[qid] >= k :
                continue
            qrel[qid][docid] = [0]
            non_relevant_docs[qid] += 1

    for qid in qrel:
        total_nonrel = len([doc for doc in qrel[qid] if qrel[qid][doc][0]==0])
        assert total_nonrel == k

    return qrel


def read_file(fname, data):
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split()
            qid, docid, score = line[0], line[2], float(line[4])
            if docid in data[qid]:
                data[qid][docid].append(score)

    return data


def get_features(data, dirpath):
    features = ['proximity', 'tf-idf', 'okapi-tf', 'okapi_bm25', 'laplace', 'lm-jm']
    for file in features:
        fname = dirpath + file + ".out"
        print(f"Reading file : {fname}")
        data = read_file(fname, data)

    for qid in data.copy():
        for doc in data[qid].copy():
            length = len(data[qid][doc])
            if length != (len(features)+1):
                print(f"Deleted doc {doc} from {qid}")
                del data[qid][doc]
    return data


def split_data(qids, i, n=5):
    start = n*i
    end = (n*i) + 5
    test = qids[start:end]
    train = qids[0:start] + qids[end:]
    return train, test


def create_model(data, keys, model_type):
    x, y = [], []
    for key in keys:
        for doc in data[key]:
            x.append(data[key][doc][1:])
            y.append(data[key][doc][0])

    x = np.array(x)
    y = np.array(y)

    if model_type == "reg":
        model = LinearRegression()
    elif model_type == "tree":
        model = tree.DecisionTreeClassifier()
    elif model_type == 'svm':
        model = svm.SVC()
    elif model_type == "logit":
        model = LogisticRegression(random_state=0)
    else:
        raise Exception("Model type not defined")

    model = model.fit(x, y)
    print(f"Training loss : {model.score(x, y)}")
    return model


def fit_model(model, data, keys):
    x = []
    result = []
    for key in keys:
        for doc in data[key]:
            x.append(data[key][doc][1:])
            result.append([key, doc])

    predicted_rel = model.predict(np.array(x))
    output = {}
    for idx, val in enumerate(predicted_rel):
        qid, docid = result[idx][0], result[idx][1]
        if qid in output:
            output[qid].append([docid, val])
        else:
            output[qid] = [[docid, val]]

    return output


def write_to_file(result, fname):
    with open(fname, "w") as fout:
        for key in result:
            res = sorted(result[key], key=operator.itemgetter(1), reverse=True)
            for idx, res in enumerate(res[:1000]):
                # qid, "Q0", doc, rank, score, "ES"
                final_input = f"{key} Q0 {res[0]} {idx+1} {res[1]} ES \n"
                fout.write(final_input)



def main(args):
    queries = read_queries(args.queries)  # Get the query numbers
    qrel = read_qrel(args.qrel, queries.keys())  # Read qrel file and get the relevant docs matching with query numbers
    data = read_additional(args.rfile, qrel, args.k)  # Read the IR function output file to get the non-relevant docs
    data = get_features(data, args.dirpath) # Creates feature dictionary from the previously run models output files
    qids = list(data.keys())
    random.shuffle(qids)
    for i in range(5):
        train_keys, test_keys = split_data(qids, i)

        model = create_model(data, train_keys, args.model)

        result_train = fit_model(model, data, train_keys)
        write_to_file(result_train, os.path.join(args.result, "train_%d_"%i + args.model+ ".txt"))

        result_test = fit_model(model, data, test_keys)
        write_to_file(result_test, os.path.join(args.result, "test_%d_"%i + args.model+ ".txt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dirpath", type=str, default="../HW2/output/models/stemmed/", help="")  # rankedFile
    parser.add_argument("--rfile", type=str, default="../HW2/output/models/stemmed/okapi_bm25.out", help="") #rankedFile
    parser.add_argument("--qrel", type=str, default="data/qrel.txt", help="")
    parser.add_argument("--result", type=str, default="output/result", help="")
    parser.add_argument("--queries", type=str, default="data/queryfile.txt", help="")
    parser.add_argument("--model", type=str, default="reg", help="")
    parser.add_argument("--k", type=int, default=1000, help="")
    args = parser.parse_args()
    main(args)