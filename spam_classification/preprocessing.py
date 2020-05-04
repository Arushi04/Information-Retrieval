import argparse
import os
from os.path import join as fjoin
import tqdm
import random

from bs4 import BeautifulSoup
import email
import spacy

from elasticsearch import Elasticsearch

import pickle

nlp = spacy.load("en_core_web_sm")
VOCAB = spacy.load("en_core_web_md").vocab.strings


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

            },
            "mappings": {
                "properties": {
                    "subject": {
                        "type": "text",
                        "fielddata": True,
                        "index_options": "positions"
                    },
                    "text": {
                        "type": "text",
                        "fielddata": True,
                        "index_options": "positions"
                    },
                    "spam": {
                        "type": "text",
                        "fielddata": True,
                        "store": True
                    },
                    "split": {
                        "type": "text",
                        "fielddata": True,
                        "store": True
                    }
                }
            }
        }
        print("index created")
        es.indices.create(index=index, body=request_body, ignore=400)


def store_in_ES(index, data, labels, train, test, es):
    for key in tqdm.tqdm(data):
        subject, content = data[key][0], data[key][1]
        if key in labels['spam']:
            spam = 'yes'
        else:
            spam = 'no'

        if key in train['spam'] or key in train['ham']:
            split = 'train'
        if key in test['spam'] or key in test['ham']:
            split = 'test'

        doc = {
            'head': subject,
            'text': content,
            'spam': spam,
            'split': split
        }
        es.index(index=index, id=key, body=doc)


def post_process(text):
    doc = nlp(text, disable=["parser", "tagger", "ner"])
    clean_text = [token.orth_ for token in doc if not token.is_punct | token.is_space | token.is_stop]
    clean_text = [token.lower() for token in clean_text if token.lower() in VOCAB]
    return " ".join(clean_text)


def get_text_from_html(text):
    html_text = str(text)
    bs = BeautifulSoup(html_text, 'html.parser')
    text = bs.get_text().strip()
    text = text.replace("\n", " ")
    return text


def get_text_from_email(text):
    body = email.message_from_string(text)
    subject = ''
    if body['subject'] is not None:
        subject = body['subject']

    body_text = ''
    if body.is_multipart():
        for part in body.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition'))

            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                body_text += part.get_payload()
            elif content_type =='text/html' and 'attachment' not in content_disposition:
                html_text = part.get_payload()
                parsed_text = get_text_from_html(html_text)
                body_text += parsed_text
    else:
        content_type = body.get_content_type()
        content_disposition = str(body.get('Content-Disposition'))
        if content_type == 'text/plain' and 'attachment' not in content_disposition:
            body_text += body.get_payload()
        elif content_type == 'text/html' and 'attachment' not in content_disposition:
            html_text = body.get_payload()
            parsed_text = get_text_from_html(html_text)
            body_text += parsed_text
    subject = post_process(subject)
    body_text = post_process(body_text)
    return subject, body_text


def read_labels(fname):
    '''
    spam ../data/inmail.179
    '''
    labels = {}
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split("/")
            label = line[0].split(" ")[0]
            file = line[2]
            if label in labels:
                labels[label].append(file)
            else:
                labels[label] = [file]
    return labels


def read_data(dir, savepath=''):
    data = {}

    if os.path.isfile(savepath):
        with open(savepath, "rb") as handle:
            data = pickle.load(handle)
        return data

    files = os.listdir(dir)
    for file in tqdm.tqdm(files):
        path = fjoin(dir, file)
        text = open(path, 'r', encoding='ISO-8859-1').read()
        subject, body_text = get_text_from_email(text)
        data[file] = [subject, body_text]

    with open(savepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data


def main(args):
    random.seed(args.seed)
    print("starting to read the data")
    data = read_data(args.dirpath, args.savepath)   # {mail.1 : <content>, mail.2 : <content>}
    print("read the data")
    labels = read_labels(args.labels)  # {spam : [mail.1, mail.2], ham:[mail.3]}
    print("read the labels")
    train, test = {}, {}  #train -> {spam : [], ham:[]}, test -> {spam : [], ham:[]}
    for key in labels:
        ratio = int(0.8 * len(labels[key]))
        random.shuffle(labels[key])
        train[key] = labels[key][:ratio]
        test[key] = labels[key][ratio:]
    print("train and test done")
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    if es.indices.exists(args.index):
        print("Index and data already exists on ES")
    else:
        create_index(args.index, es)
        print("data is getting stored on ES")
        store_in_ES(args.index, data, labels, train, test, es)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dirpath", type=str, default="data/trec/data/", help="")
    parser.add_argument("--labels", type=str, default="data/trec/full/index", help="")
    parser.add_argument("--savepath", type=str, default="data/data.pickle", help="")
    parser.add_argument("--index", type=str, default="spam_data", help="")
    parser.add_argument("--seed", type=int, default=4, help="")

    args = parser.parse_args()
    main(args)