import argparse
from os.path import join as fjoin
import pickle
import os
from os.path import isfile

fo = open("inlinkgraphs.txt", "a")

def write_to_graph(url, inlinks, count):
    print("writing the url no : ", count, url)
    line = "%s %s\n" % (url, " ".join(inlinks))
    fo.write(line)


def main(args):
    checkpoint_path = fjoin(args.ckp, "checkpoint.%d." % args.ckp_no)

    if isfile(checkpoint_path + "frontier_map.pt"):
        frontier_map = pickle.load(open(checkpoint_path + "frontier_map.pt", "rb"))
    else:
        raise Exception("checkpoint not found")
    count = 0
    for file in os.listdir(args.cdp):
        path = fjoin(args.cdp, file)
        res = pickle.load(open(path, "rb"))
        url = res['docno']
        inlinks = list(frontier_map[url].inlinks)
        count += 1
        write_to_graph(url, inlinks, count)
    fo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir", type=str, default="./output/", help="")
    parser.add_argument("--ckp_no", type=int, default=40000, help="")

    args = parser.parse_args()

    # additional parse option
    args.cdp = fjoin(args.dir, "crawled") #cdp = crawled data path
    args.ckp = fjoin(args.dir, "checkpoint")  # ckp = checkpoint
    main(args)
