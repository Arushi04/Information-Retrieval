from __future__ import division
import argparse
import pickle
import random
import math
import operator
import matplotlib.pyplot as plt


def add_outlinks(root_set, merged_outlinks, base_set):
    for url in root_set:
        if url in merged_outlinks:
            outlinks = merged_outlinks[url]
            base_set = base_set.union(outlinks)
    print("length of baseset after adding outlinks: ", len(base_set))
    return base_set


def add_inlinks(root_set, merged_inlinks, base_set, d, max_size):
    for url in root_set:
        if len(base_set) > max_size:
            break
        if url in merged_inlinks:
            inlinks = merged_inlinks[url]
            if len(inlinks) > d:
                inlinks = random.sample(list(set(inlinks)), d)

            base_set = base_set.union(set(inlinks))

    print("length of base set after adding inlinks: ", len(base_set))
    return base_set


def read_rootset(fname, merged_inlinks, merged_outlinks, D=100, MAX_SIZE=10000):
    root_set = set()
    base_set = set()
    with open(fname, "r") as fr:
        for line in fr:
            line = line.strip().split()
            url = line[0]
            root_set.add(url)
            base_set.add(url)

    print("Base_set in read: ", len(base_set))
    base_set = add_outlinks(root_set, merged_outlinks, base_set)
    base_set = add_inlinks(root_set, merged_inlinks, base_set, D, MAX_SIZE)
    return root_set, base_set


def save_to_file(sorted_list, fname):
    with open(fname, "w") as fout:
        for t in sorted_list:
            fout.write(str(t) + "\n")


def main(args):
    M = pickle.load(open(args.pickles[0], "rb"))  # INLINK
    P = pickle.load(open(args.pickles[1], "rb"))  # OUTLINK

    authority, hub = {}, {}
    A_norm, H_norm = [], []
    eps = 1e-8
    prev_a_norm, prev_h_norm = None, None
    root_set, base_set = read_rootset(args.input, M, P, args.d, args.max_size)

    # initialize authority and hub_scores
    for page in base_set:  # base_set / root_set
        authority[page] = 1.0
        hub[page] = 1.0

    idx = 1
    while True:
        norm = 0
        for p in base_set:
            authority[p] = 0
            if p in M:  # verify
                for q in M[p]:  # in-link
                    if q in hub:
                        authority[p] += hub[q]
            norm += math.pow(authority[p], 2)
        norm = math.sqrt(norm)

        for p in base_set:
            authority[p] = authority[p] / norm
        a_norm = norm

        norm = 0
        for p in base_set:
            hub[p] = 0
            if p in P:  # verify
                for r in P[p]:  # out-link
                    if r in authority:
                        hub[p] += authority[r]
            norm += math.pow(hub[p], 2)
        norm = math.sqrt(norm)

        for p in base_set:
            hub[p] = hub[p] / norm
        h_norm = norm

        A_norm.append(a_norm)
        H_norm.append(h_norm)

        print(f"iter: {idx} :: a_norm : {a_norm} and h_norm : {h_norm}")

        a_converge = False
        if prev_a_norm is not None and abs(a_norm - prev_a_norm) <= eps:
            a_converge = True

        if a_converge and prev_h_norm is not None and abs(h_norm - prev_h_norm) <= eps:
            break

        prev_a_norm = a_norm
        prev_h_norm = h_norm
        idx += 1

    sorted_A = sorted(authority.items(), key=operator.itemgetter(1), reverse=True)
    sorted_H = sorted(hub.items(), key=operator.itemgetter(1), reverse=True)

    for t in sorted_A[0:5]:
        print(f"\t Authority {t}")

    for h in sorted_H[0:5]:
        print(f"\t Hub {h}")

    save_to_file(sorted_A[0:500], args.outfiles[0])
    save_to_file(sorted_H[0:500], args.outfiles[1])
    plt.plot([i + 1 for i in range(len(A_norm))], A_norm)
    plt.plot([i + 1 for i in range(len(H_norm))], H_norm)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("-i", "--input", type=str, default="output/esbuiltin.txt", help="")
    parser.add_argument('-p', '--pickles', nargs='+', action='store',
                        default=["merged_inlink.pickle", "merged_oulink.pickle", "merged_sink.pickle"])

    parser.add_argument('-o', '--outfiles', nargs='+', action='store', help='<Required> Set flag',
                        default=["output/authority.txt", "output/hub.txt"])
    parser.add_argument('-d', '--d', type=int, default=250, help='')
    parser.add_argument('-m', '--max_size', type=int, default=10000, help='')
    args = parser.parse_args()
    print(args)
    main(args)