import argparse
import math
import matplotlib.pyplot as plt


def write_in_details(relevance, retrieved, rel_ret, R_precision, avg_precision, nDCGs, precision_k, recall_k, F1_k, q_flag=False):
    k = [5, 10, 20, 50, 100]
    count_ret, count_rel, rel_retr = 0, 0, 0
    sum_r_precision, sum_avg_precision, sum_ndcg = 0, 0, 0
    if q_flag:
        for key in R_precision:
            print(f"Query Id   :   {key}")
            print(f"\tRetrieved : {len(retrieved[key])}")
            print(f"\tRelevant : {len(relevance[key])}")
            print(f"\tRel Ret : {rel_ret[key]}")
            print(f"\t\tR-precision for Queryid {key} : {R_precision[key]}")
            print(f"\t\tAvg-precision for Queryid {key} : {avg_precision[key]}")
            print(f"\t\tnDCG for Queryid {key} : {nDCGs[key]}")

            for i in range(len(k)):
                print(f"\t\tPrecision at {k[i]} is {precision_k[key][i]}")
            for i in range(len(k)):
                print(f"\t\tRecall at {k[i]} is {recall_k[key][i]}")
            for i in range(len(k)):
                print(f"\t\tF1 measure at {k[i]} is {F1_k[key][i]}")

    for key in retrieved:
        count_ret += len(retrieved[key])
        count_rel += len(relevance[key])
        rel_retr += rel_ret[key]

    avg_r_precision = sum(R_precision.values()) / (len(R_precision))
    mean_avg_precision = sum(avg_precision.values()) / (len(avg_precision))
    mean_ndcg = sum(nDCGs.values()) / len(nDCGs)
    print(f"Total no of documents over all queries")
    print(f"Retrieved : {count_ret}")
    print(f"Relevant : {count_rel}")
    print(f"Rel_ret : {rel_retr}")
    print(f"\n\n\tR-precision for Queries are {avg_r_precision}")
    print(f"\tAvg-precision for Queries are {mean_avg_precision}")
    print(f"\tnDCG for Queries are {mean_ndcg}")



def read_qrel(fname):
    '''
    99 0 AP891222-0227 0
    99 0 AP891228-0070 0
    99 0 AP891229-0182 0
    '''
    relevance = {}
    url_rel = {}  # {99 : {AP891222-0227 : 0, AP891228-0070 : 0 }
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split()
            key, url, rel = line[0], line[2], line[3]
            if key not in url_rel:
                url_rel[key] = {url : rel}
            else:
                url_rel[key][url] = rel

            if rel == '1' or rel == '2':
                if key not in relevance:
                    relevance[key] = {url}
                else:
                    relevance[key].add(url)
    print("length of relevant docs: ", len(relevance[key]))
    return relevance, url_rel


def read_esbuiltin(fname, relevance):
    '''
    99 Q0 AP890221-0111 830 10.676816 Exp
    99 Q0 AP890215-0175 831 10.667194 Exp
    99 Q0 AP890706-0170 832 10.658821 Exp
    '''

    retrieved = {}
    count =0
    with open(fname, "r") as f:
        for line in f:
            line = line.strip().split()
            key, url, rank = line[0], line[2], line[3]
            lab = 'nr'
            if key not in retrieved:
                retrieved[key] = []

            if url in relevance[key]:
                lab = 'r'
                count += 1
            retrieved[key].append([rank, url, lab])
    return retrieved


def get_avg_precision(relevance, retrieved):
    avg_precision = {}
    rel_ret = {}
    for key in retrieved:
        REL = len(relevance[key])
        precision_sum = 0
        count = 0
        idx = 0
        for rank, url, status in retrieved[key]:
            idx+=1
            if idx > 1000:
                break
            if status == 'r':
                count += 1 #checking no of relevant docs
                precision_sum += (count/idx)
        avg_precision[key] = precision_sum / REL
        rel_ret[key] = count
    return avg_precision, rel_ret


def get_r_precision(relevance, retrieved):
    R_precision = {}
    for key in retrieved:
        R = len(relevance[key])
        rel_retrieved = 0
        for rank, url, status in retrieved[key]:
            if int(rank) > R:
                break
            if status == 'r':
                rel_retrieved += 1
        R_precision[key] = rel_retrieved / R
    #print("R precision of each query : ", R_precision)
    return R_precision


def get_nDCG(url_rel, rfile, k=200):
    # url_rel - {queryid :{url:relevance}}
    dcg = {}
    with open(rfile, "r") as f:
        for line in f:
            line = line.strip().split()
            key, url = line[0], line[2]
            rel = 0
            if url in url_rel[key]:
                rel = int(url_rel[key][url])

            if key not in dcg:
                dcg[key] = [rel]
            else:
                dcg[key].append(rel)

    nDCG = {}
    for key in dcg:
        values = dcg[key]
        sorted_values = sorted(values, reverse=True)
        sum_dcg = values[0]
        sum_ndcg_deno = sorted_values[0]
        sum_ndcg = None
        for idx in range(1, len(values)):
            val = values[idx]
            sorted_val = sorted_values[idx]
            sum_dcg += (val/math.log(idx+1))
            sum_ndcg_deno += (sorted_val/math.log(idx + 1))

        if sum_ndcg_deno:
            sum_ndcg = sum_dcg/sum_ndcg_deno
        nDCG[key] = sum_ndcg
    #print(f"DCG/nDCG for query {key} is : {sum_dcg}/{sum_ndcg}")
    return nDCG


def get_precision_k(retrieved):
    precision = {}
    k = [5, 10, 20, 50, 100]
    precision_val = []
    #rel_ret= {}
    for key in retrieved:
        for i in k:
            rel_retrieved = 0
            for rank, url, status in retrieved[key]:
                if int(rank) > i:
                    break
                if status == 'r':
                    rel_retrieved += 1
            precision_val.append(rel_retrieved/i)
        precision[key] = precision_val
        precision_val = []
    #print("precision of each query : ", rel_ret)
    return precision


def get_recall_k(retrieved, relevance):
    recall = {}
    k = [5, 10, 20, 50, 100]
    recall_val = []
    for key in retrieved:
        R = len(relevance[key])
        for i in k:
            rel_retrieved = 0
            for rank, url, status in retrieved[key]:
                if int(rank) > i:
                    break
                if status == 'r':
                    rel_retrieved += 1
            recall_val.append(rel_retrieved/R)
        recall[key] = recall_val
        recall_val = []
    return recall


def get_F1_k(precision, recall, retrieved, eps=1e-8):
    F1 = {}
    k = [5, 10, 20, 50, 100]
    F_val = []
    for key in retrieved:
        for i in range(len(k)):
            f_measure = 2 * ((precision[key][i] * recall[key][i])/(precision[key][i] + recall[key][i] + eps))
            F_val.append(f_measure)
        F1[key] = F_val
        F_val = []
    return F1


def plot_pr(precisions, recalls):
    for query in precisions:
        plt.plot(recalls[query], precisions[query], lw=2, label='query {}'.format(query))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()


def main(args):
    relevance, url_rel = read_qrel(args.qrel)  #Relevance - key:queryId, value:set of relevant urls,
    retrieved = read_esbuiltin(args.rfile, relevance)
    R_precision = get_r_precision(relevance, retrieved)
    avg_precision, rel_ret = get_avg_precision(relevance, retrieved)
    nDCGs = get_nDCG(url_rel, args.rfile)
    precision_k= get_precision_k(retrieved)
    recall_k = get_recall_k(retrieved, relevance)
    F1_k = get_F1_k(precision_k, recall_k, retrieved)

    write_in_details(relevance, retrieved, rel_ret, R_precision,  avg_precision, nDCGs, precision_k, recall_k, F1_k, args.q)
    plot_pr(precision_k, recall_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--rfile", type=str, default="output/hw5_builtin.txt", help="") #rankedFile
    parser.add_argument("--qrel", type=str, default="output/hw5_qrel.txt", help="")
    parser.add_argument("--q", action='store_true')
    args = parser.parse_args()
    main(args)