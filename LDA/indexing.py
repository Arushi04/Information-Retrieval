import os
import io
from elasticsearch import Elasticsearch
import tqdm
import pandas as pd
from datetime import datetime
from elasticsearch import helpers

es = Elasticsearch([{'host':'localhost','port':9200}])
dir_path = "/Users/arushi/PycharmProjects/InformationRetrieval/Information-Retrieval/data/AP_DATA/ap89_collection"
INDEX_NAME = "ap_dataset1"


request_body = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "max_result_window": "85000",
        },
    "mappings": {
        "properties": {
            "text": {
                "type": "text",
                "fielddata": True,
                "index_options": "positions"
            }
        }
    }
}
request = es.indices.create(index=INDEX_NAME, body=request_body, ignore=400)
print(request)

for fname in os.listdir(dir_path):
    if not fname.startswith("ap"):
        continue

    current_docid = ""
    current_text = []
    text_tag_started = False
    filepath = os.path.join(dir_path, fname)
    with io.open(filepath, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.strip()
            if line.startswith("<DOCNO>"):
                current_docid = line.split("<DOCNO>")[1].split("</DOCNO>")[0].strip()
                continue

            if line.startswith("<TEXT>"):
                text_tag_started = True
                continue

            if line.startswith("</TEXT>"):
                text_tag_started = False
                continue

            if text_tag_started:
                current_text.append(line)

            if line.startswith("</DOC>"):
                if current_docid != "":
                    current_text = " ".join(current_text)

                    doc = {
                        'text': current_text
                    }

                    res = es.index(index=INDEX_NAME, id=current_docid, body=doc)
                    current_text = []
