import pickle
from elasticsearch import Elasticsearch
import os
import argparse
from os.path import isfile
from os.path import join as fjoin
from urlDetails import URL
import json

def create_index(index, es):
    print("inside creating index")
    if es.indices.exists(index):
        print("Index already exists")
    else:
        request_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "max_result_window": "85000",
                "analysis": {
                    "filter": {
                        "english_stop": {
                            "type": "stop",
                            "stopwords_path": "my_stoplist.txt"
                        },

                        "my_stemmer": {
                            "type": "stemmer",
                            "name": "english"
                        }
                    },

                    "analyzer": {
                        "stopped": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase", "english_stop", "my_stemmer"
                            ]
                        }
                    },

                }
            },
            "mappings": {
                "properties": {
                    #"id": {  # url
                    #   "type": "text",
                    #    "store": True,
                    #    "index": "stopped",
                    #    "term_vector": "with_positions_offsets_payloads"
                    #},
                    "head": {  # title
                        "type": "text",
                        "fielddata": True,
                        "analyzer": "stopped",
                        "index_options": "positions"
                    },
                    #"httpheader": {
                    #    "type": "text",
                    #    "store": True
                    #},
                    "text": {
                        "type": "text",
                        "fielddata": True,
                        "analyzer": "stopped",
                        "index_options": "positions"
                    },
                    #"rawhtml": {
                    #    "type": "text",
                    #    "store": True
                    #},
                    "outlinks": {
                        "type": "text",
                        "fielddata": True,
                        "store": True
                    },
                    "inlinks": {
                        "type": "text",
                        "fielddata": True,
                        "store": True
                    }
                }
            }
        }
        es.indices.create(index=index, body=request_body, ignore=400)


def store_in_ES(index, url, title, content,inlinks, outlinks, es):
    """Store an id, the URL, the HTTP headers, the page contents cleaned (with term positions),
     the raw html, and a list of all in-links (known) and out-links for the page."""
    #print("storing")

    doc = {
        'head': title,
        'text': content,
        'inlinks': inlinks,
        'outlinks': outlinks
    }
    res = es.index(index=index, id=url, body=doc)


def main(args):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    #create the settings and mapping of the index
    create_index(args.index, es)

    checkpoint_path = fjoin(args.ckp, "checkpoint.%d." % args.ckp_no)

    if isfile(checkpoint_path + "frontier.pt"):

        frontier = pickle.load(open(checkpoint_path + "frontier.pt", "rb"))
        frontier_map = pickle.load(open(checkpoint_path + "frontier_map.pt", "rb"))
        id_to_url = pickle.load(open(checkpoint_path + "id_to_url.pt", "rb"))
        links_crawled = pickle.load(open(checkpoint_path + "links_crawled.pt", "rb"))
        current_wave = pickle.load(open(checkpoint_path + "current_wave.pt", "rb"))
    else:
        raise Exception("checkpoint not found")


    #Load all the pickles of the crawled data
    for file in os.listdir(args.cdp):
        path = fjoin(args.cdp, file)
        res = pickle.load(open(path, "rb"))

        url = res['docno']
        title = res['head']
        content = res['text']
        inlinkData = list(frontier_map[url].inlinks)
        outlinkData = list(frontier_map[url].outlinks)
        print("inlink data : ", inlinkData)
        inlinks = json.dumps(inlinkData)
        outlinks = json.dumps(outlinkData)
        print("inlinks after json dumping : ", inlinks)
        store_in_ES(args.index, url, title, content, inlinks, outlinks, es)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir", type=str, default="./output/", help="")
    parser.add_argument("--index", type=str, default="church_data", help="")
    parser.add_argument("--ckp_no", type=int, default=40000, help="")
    args = parser.parse_args()

    # additional parse option
    args.cdp = fjoin(args.dir, "crawled") #cdp = crawled data path
    args.ckp = fjoin(args.dir, "checkpoint")  # ckp = checkpoint
    main(args)
