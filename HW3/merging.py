import elasticsearch
from elasticsearch import Elasticsearch
import elasticsearch.helpers
import argparse
import os
from os.path import join as fjoin
from os.path import isfile
import pickle
import logging
import sys
import json


def merge_inlinks(inlinks_list):
    results_union = set().union(*inlinks_list)
    return list(results_union)


def main(args):
    #es2 = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    es2 = Elasticsearch("https://96aa4157ead74b5ca4926523b1d1994e.us-east-1.aws.found.io:9243",
                                http_auth=('elastic', 'MrkfJ5hxIcCOzTMfOa1Nftzy'))

    #elasticsearch.helpers.reindex(es1, "church_data", args.out_index, query=None, target_client=None,
     #           chunk_size=500, scroll='5m', scan_kwargs={}, bulk_kwargs={})

    checkpoint_path = fjoin(args.ckp, "checkpoint.%d." % args.ckp_no)
    if isfile(checkpoint_path + "frontier_map.pt"):
        frontier_map = pickle.load(open(checkpoint_path + "frontier_map.pt", "rb"))
    else:
        raise Exception("checkpoint not found")

    filesadded = 0
    filesupdated = 0
    # Load all the pickles of the crawled data
    for file in os.listdir(args.cdp):
        path = fjoin(args.cdp, file)
        res = pickle.load(open(path, "rb"))
        url = res['docno']

        inlinkData = list(frontier_map[url].inlinks)
        j_inlinks = json.dumps(inlinkData)
        logging.info("Checking for url {}".format(url))

        #Finding if the url is in the merged index
        result = es2.get(index=args.out_index, id=url, ignore=404)

        if result['found'] is True:
            logging.info("inlinks from local  {}".format(len(set((frontier_map[url].inlinks)))))
            logging.info("inlinks retrieved {}".format(len(set(result['_source']['inlinks']))))
            existing_inlinks = json.loads(j_inlinks)
            retrieved_inlinks = json.loads(result['_source']['inlinks'])

            #merging the inlinks from both local and merged set and updating the inlinks
            final_inlinkset = merge_inlinks([retrieved_inlinks, existing_inlinks])
            logging.info("length of final list {}".format(len(final_inlinkset)))
            es2.update(index=args.out_index, id=url, doc_type=args.doc_type, body={"doc": {"inlinks":json.dumps(final_inlinkset)}})
            filesupdated += 1
            logging.info("doc updated for url {}".format(url))

        else:
            # indexing the data for the url which doesn't match any url in merged data index
            logging.info("value of res in else {}: ".format(len(result)))
            title = res['head']
            content = res['text']
            inlinks = j_inlinks
            outlinkData = list(frontier_map[url].outlinks)
            outlinks = json.dumps(outlinkData)
            doc = {
                'head': title,
                'text': content,
                'inlinks': inlinks,
                'outlinks': outlinks
            }
            es2.index(index=args.out_index, id=url, body=doc, doc_type=args.doc_type)
            filesadded += 1
            logging.info("doc added for url {}: ".format(url))

    logging.info("doc added {} and updated {}: ".format(filesadded, filesupdated))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir", type=str, default="./output/", help="")
    parser.add_argument('-out', '--out_index', type=str, default="ir_hw03_prod2", help="")
    parser.add_argument('-doc', '--doc_type', type=str, default="hw03_crawl_contents", help="")
    parser.add_argument("--ckp_no", type=int, default=40000, help="")
    args = parser.parse_args()
    args.logfile = os.path.join(args.dir, "log_merging.txt")

    logging.basicConfig(filename=args.logfile, format='%(asctime)s %(message)s', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    args.cdp = fjoin(args.dir, "crawled")  # cdp = crawled data path
    args.ckp = fjoin(args.dir, "checkpoint")  # ckp = checkpoint
    main(args)
