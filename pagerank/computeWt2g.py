from __future__ import division
import argparse
import math
import operator
import matplotlib.pyplot as plt

def read_inlinksgraph(fname):
    P = {}  # dict of outlinks
    S = set()
    M = {}  # dict of inlinks
    with open(args.fname, "r") as f:
        for line in f:
            line = line.strip().split(" ")
            root_url, incoming_links = line[0], line[1:]
            M[root_url] = set(incoming_links)
            for inlink in incoming_links:
                if inlink not in M:
                    M[inlink] = {}
                if inlink not in P:
                    P[inlink] = {root_url}
                else:
                    P[inlink].add(root_url)
    for page in M:
        if page not in P:
            P[page] = {}
        if len(P[page]) == 0:
            S.add(page)

    assert sorted(P.keys()) == sorted(M.keys())
    return P, S, M


def get_perplexity(vals):
    sum = 0
    for val in vals:
        prob = val * math.log(val, 2)
        sum += prob
    ppl = math.pow(2, -sum)
    return ppl


def save_to_file(sorted_pr, fname, P, M):
    fo = open(fname, "a")
    print("writing to file ", fname)
    for s_tuple in sorted_pr:
        url = s_tuple[0]
        inlink_count = len(M[url])
        outlink_count = len(P[url])
        line = "%s %d %d \n" % (s_tuple, inlink_count, outlink_count)
        fo.write(line)
    fo.close()

def main(args):
    '''
    // P is the set of all pages; |P| = N
    // S is the set of sink nodes, i.e., pages that have no out links
    // M(p) is the set of pages that link to page p
    // L(q) is the number of out-links from page q
    // d is the PageRank damping/teleportation factor; use d = 0.85 as is typical
    '''

    P, S, M = read_inlinksgraph(args.fname)

    N = len(P)
    d = 0.85
    PR = {}
    eps = 1e-3
    iter = 1
    prev_ppl = None
    all_ppl = []

    for p in P:
        PR[p] = 1/N
    while True:
        sinkPR = 0
        newPR = {}
        for p in S:
            sinkPR += PR[p]
        for p in P:
            newPR[p] = (1-d)/N
            newPR[p] += (d * sinkPR)/N
            for q in M[p]:
                newPR[p] += (d * PR[q]) / len(P[q])
        for p in P:
            PR[p] = newPR[p]

        sorted_pr = sorted(PR.items(), key=operator.itemgetter(1), reverse=True)
        for s_tuple in sorted_pr[0:5]:
            print("\t", s_tuple)

        ppl = get_perplexity(PR.values())
        print("Iter : {} and Perplexity : {} ".format(iter, ppl))
        all_ppl.append(ppl)

        if prev_ppl is not None and abs(ppl - prev_ppl) <= eps:
            break
        prev_ppl = ppl
        iter += 1
    save_to_file(sorted_pr[0:500], args.outfile, P, M)
    plt.plot([i+1 for i in range(len(all_ppl))], all_ppl)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--fname", type=str, default="wt2g_inlinks.txt", help="")
    parser.add_argument("--outfile", type=str, default="output/wt2g_res.txt", help="")
    parser.add_argument("--outgraph", action="store_true", help="Flag to do something")
    args = parser.parse_args()
    main(args)