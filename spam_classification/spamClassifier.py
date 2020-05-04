import argparse
from elasticsearch import Elasticsearch
import os
import operator
import pickle
import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix


def read_features(fname, cutoff=None):
    features = []
    with open(fname, "r") as f:
        idx = 1
        for line in f:
            word = line.strip().split()[0]
            features.append(word)
            if cutoff is not None and idx>=cutoff:
                break
            idx += 1

    return features


def get_train_test(es, index):
    query_body = {"query": {"bool": {"must": {"match": {"split": "train"}}}}}
    result = es.search(index=index, body=query_body, size=85000)
    data = [doc for doc in result['hits']['hits']]
    train_data, test_data = [], []
    for doc in data:
        train_data.append(doc['_id'])

    query_body = {"query": {"bool": {"must": {"match": {"split": "test"}}}}}
    result = es.search(index=index, body=query_body, size=85000)
    data = [doc for doc in result['hits']['hits']]
    for doc in data:
        test_data.append(doc['_id'])

    print("length of train : ", len(train_data))
    print("length of test : ", len(test_data))
    return train_data, test_data


def divide_chunks(l, n=250):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_matching_docs_from_ES(es, index, feature):
    data = []
    query_body = {"query": {"bool": {"must": {"match": {"text": feature}}}}}
    results = es.search(index=index, body=query_body, size=85000)['hits']['hits']
    results = [result['_id'] for result in results]
    print("Total docs retrieved:%d" % len(results))

    results_chunk = divide_chunks(results)
    for results in results_chunk:
        mterms = es.mtermvectors(index=index, doc_type='_doc',
                                 body=dict(ids=results,
                                           parameters=dict(term_statistics=True,
                                                           fields=['text'])))['docs']

        for mterm in mterms:
            if "text" in mterm['term_vectors']:
                if feature in mterm['term_vectors']['text']['terms']:
                    doc_id = mterm["_id"]
                    tf = mterm['term_vectors']['text']['terms'][feature]['term_freq']
                    data.append([doc_id, tf])
    return data


def create_label_matrix(fname, index2doc):
    doc2label = {}
    with open(fname, 'r') as f:
        for line in f:
            '''spam ../data/inmail.1'''
            line = line.strip().split()
            label = line[0]
            file = line[1].split("/")[-1]
            if label == 'spam':
                doc2label[file] = 1
            else:
                doc2label[file] = 0
    matrix = [doc2label[index2doc[i]] for i in range(len(index2doc))]
    return matrix


def create_model(data, doc_ids, labels, model_type, is_sparse, feas):
    if not is_sparse:
        x, y = [], []

        for doc in doc_ids:
            idx = doc_ids[doc]
            x.append(data[idx])
            y.append(labels[idx])
        x = np.array(x)
        y = np.array(y)
    else:
        x = data
        y = labels
    print(f"{model_type}")

    if model_type == "reg":
        model = LinearRegression()
    elif model_type == "logit":
        model = LogisticRegression(solver="liblinear", max_iter=1500)
    elif model_type == "tree":
        model = tree.DecisionTreeClassifier()
    elif model_type == 'nb':
        model = MultinomialNB()
    else:
        raise Exception("Model type not defined")

    model = model.fit(x, y)
    if model_type == "logit":
        coef_weights = model.coef_.tolist()[0]
        feas = list(zip(feas, coef_weights))
        feas = sorted(feas, key=operator.itemgetter(1), reverse=True)
        for iw in feas[:30]:
            print(iw)

    print(f"Score : {model.score(x, y)}")
    print(f" model class :  {model.classes_}")
    return model


def fit_model(model, data, labels, doc2idx, is_sparse):
    output = {}
    if not is_sparse:
        x, y = [], []
        for doc in doc2idx:
            idx = doc2idx[doc]
            x.append(data[idx])
            y.append(labels[idx])
        x = np.array(x)
    else:
        x = data
        y = labels

    probability = model.predict_proba(x)[:, 1]

    for doc in doc2idx:
        output[doc2idx[doc]] = [doc]

    y_true = np.array(y)
    y_scores = np.array(probability)
    roc_score = roc_auc_score(y_true, y_scores)

    for idx, val in enumerate(probability):
        output[idx].append(val)

    print("roc auc score :", roc_score)
    return output


def write_to_file(output, fname):
    with open(fname, "w") as fout:
        values = output.values()
        result = sorted(values, key=operator.itemgetter(1), reverse=True)
        for res in result:
            # doc spam_score
            final_input = f"{res[0]} {res[1]} \n"
            fout.write(final_input)


def write_file(fname, matrix, labels, is_sparse):
    with open(fname, "w") as f:
        for i in range(len(matrix)):
            if not is_sparse:
                f.write("\t".join(map(str, matrix[i])))
                f.write("\tL=%d\n" % (labels[i]))
            else:
                line = ['+1'] if labels[i] == 1 else ['-1']
                for j in matrix[i]:
                    line.append("%d:%d"%(j[0],j[1]))
                line = " ".join(line)
                f.write("%s\n"%line)


def main(args):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    features = read_features(args.features, args.cutoff)
    print(f"length of features : {len(features)}")

    train, test = get_train_test(es, args.index)

    train_doc2idx, test_doc2idx = {}, {}
    train_idx2doc, test_idx2doc = {}, {}
    fea2idx = {}
    for idx, i in enumerate(train):
        train_doc2idx[i] = idx
        train_idx2doc[idx] = i
    for idx, i in enumerate(test):
        test_doc2idx[i] = idx
        test_idx2doc[idx] = i

    for idx, i in enumerate(features):
        fea2idx[i] = idx

    with open('feature_index.txt', 'w') as outfile:
        json.dump(fea2idx, outfile, sort_keys=True, indent=4, separators=(',', ': '))

    train_label = create_label_matrix(args.labels, train_idx2doc)
    test_label = create_label_matrix(args.labels, test_idx2doc)

    if not args.sparse:
        train_matrix = [[0 for i in range(len(features))] for j in range(len(train))]
        test_matrix = [[0 for i in range(len(features))] for j in range(len(test))]

        for fdx, feature in enumerate(features):
            print(feature)
            data = get_matching_docs_from_ES(es, args.index, feature)
            for doc in data:
                if doc[0] in train_doc2idx:
                    train_idx = train_doc2idx[doc[0]]
                    train_matrix[train_idx][fdx] = doc[1]

                if doc[0] in test_doc2idx:
                    test_idx = test_doc2idx[doc[0]]
                    test_matrix[test_idx][fdx] = doc[1]
        write_file("train_feat.csv", train_matrix, train_label, args.sparse)
        write_file("test_feat.csv", test_matrix, test_label, args.sparse)
    else:
        with open("data/unigram.train.stats", "rb") as handle:
            train_stats = pickle.load(handle)
        with open("data/unigram.test.stats", "rb") as handle:
            test_stats = pickle.load(handle)
        train_sparse_f = open("data/train_sparse_fea.txt", "w")
        test_sparse_f = open("data/test_sparse_fea.txt", "w")
        feat_info_f = open("data/sparse_fea.txt", "w")

        for jdx,fea in enumerate(features):
            feat_info_f.write("%d\t%s\n"%(jdx, fea))


        train_sparse_info = []
        test_sparse_info = []
        for doc in train_stats:
            out_line = ["%s"%train_label[train_doc2idx[doc]]]
            for token in train_stats[doc]:
                if token not in fea2idx:
                    continue
                tf = train_stats[doc][token]
                rid = train_doc2idx[doc]
                cid = fea2idx[token]
                out_line.append("%d:%d"%(cid, tf))
                train_sparse_info.append([rid, cid, tf])
            train_sparse_f.write(" ".join(out_line)+"\n")
        for doc in test_stats:
            out_line = ["%s"%test_label[test_doc2idx[doc]]]
            for token in test_stats[doc]:
                if token not in fea2idx:
                    continue
                tf = test_stats[doc][token]
                rid = test_doc2idx[doc]
                cid = fea2idx[token]
                out_line.append("%d:%d" % (cid, tf))
                test_sparse_info.append([rid, cid, tf])
            test_sparse_f.write(" ".join(out_line) + "\n")

        row, col, data = zip(*train_sparse_info)
        train_matrix = csr_matrix((data, (row, col)), shape=(len(train), len(features)))
        row, col, data = zip(*test_sparse_info)
        test_matrix = csr_matrix((data, (row, col)), shape=(len(test), len(features)))

    model = create_model(train_matrix, train_doc2idx, train_label, args.model, args.sparse, features)

    result_train = fit_model(model, train_matrix, train_label, train_doc2idx, args.sparse)
    write_to_file(result_train, os.path.join(args.result, "train_" + args.model + ".txt"))

    result_test = fit_model(model, test_matrix, test_label, test_doc2idx, args.sparse)
    write_to_file(result_test, os.path.join(args.result, "test_" + args.model + ".txt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index", type=str, default="spam_data", help="")
    parser.add_argument("--labels", type=str, default="data/trec/full/index", help="")
    parser.add_argument("--savepath", type=str, default="data/data.pickle", help="")
    parser.add_argument("--features", type=str, default="data/features.txt", help="")
    parser.add_argument("--cutoff", type=int, default=50000, help="")
    parser.add_argument("--result", type=str, default="output/", help="")
    parser.add_argument("--model", type=str, default="logit", help="")
    parser.add_argument("--sparse", action='store_true')
    args = parser.parse_args()
    main(args)