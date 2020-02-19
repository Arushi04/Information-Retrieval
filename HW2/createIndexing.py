import argparse
import os
import io
import pickle

from custom_util import my_tokenize, compress_line, encode_this, decode_this
from custom_util import divide_chunks, read_stopwords, my_stemmer
'''
Running the file:
with stem : python createIndexing.py --do_stem --outf output/stem_updated
without stem : python createIndexing.py --outf output/unstemmed
with compress : python createIndexing.py --do_stem --compress --outf output/stem_compressed/
'''


def write_to_file(partial_inverted_index, index, outf, compress):
    catalog = {}
    last_offset = 0
    print("Writing partial_inverted_index_%d.txt" % index)
    with io.open("%s/partial_inverted_index_%d.txt" % (outf, index), "wb") as f:
        for term in partial_inverted_index:
            ts = partial_inverted_index[term]["ts"]
            fs = partial_inverted_index[term]["fs"]
            ttf = fs["ttf"]
            line = ["%d %s" % (k, ",".join(map(str, ts[k]))) for k in ts]
            line = " ".join(line)
            line = "%s %d %s\n" % (term, ttf, line) # term ttf docid1 pos1 docid2 pos1,pos2 docid3
            line = compress_line(line, compress)
            catalog[term] = [last_offset, len(line)]
            last_offset += len(line)
            f.write(line)
    pickle.dump(catalog, open("%s/partial_catalog_%d.txt" % (outf, index), "wb"))


def save_vocab(data, fname):
    print("Writing vocab to pickle file %s/vocab.pickle" % fname)
    pickle.dump(data, open("%s/vocab.pickle" % fname, "wb"))


def read_data(path):
    """
    reads all files in the directory path. doc_data is a dictionary with current_docid as key and current_text as val
    """
    doc_data = {}
    for fname in os.listdir(path):
        if not fname.startswith("ap"):
            continue

        current_docid = ""
        current_text = []
        text_tag_started = False
        filepath = os.path.join(path,fname)
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
                        doc_data[current_docid] = current_text
                        current_text = []
    print("len Doc data %d " % len(doc_data))
    return doc_data

def tokenize_build(text, do_stem, stopwords, vocab, docid):
    """
        This function tokenizes the code and do stemming according to do_stem status(True/False)
        token_list -> (term1, 20, 1), (term2, 20, 2), (term3, 20, 3), (term4, 20, 4) [token, docid, term_position]
    """
    token_list = []
    tokens = my_tokenize(text, do_stem, stopwords)
    position = 0
    for token in tokens:
        if token not in vocab:
            token_id = len(vocab) + 1
            vocab[token] = token_id
        else:
           token_id = vocab[token]

        position += 1
        token_list.append([token, docid, position])
    return token_list, vocab


def main(args):
    # Creating output directory
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    # Reading doc data and dividing it into chunks of 1000
    doc_data = read_data(args.dir_path)
    doc_data_chunks = divide_chunks(doc_data)

    # Reading stopwords
    stopwords = read_stopwords(args.stopfile)

    # token id and docIDn
    vocab, docid_vocab = {}, {"ntoi": {}, "iton": {}}
    sum_ttf = 0
    doc_count = 0
    doc_len = {}

    for index, doc_chunk in enumerate(doc_data_chunks):  #looping each 1000 batch in total data
        partial_inverted_index = {}
        for docname in doc_chunk:   #docname : AP890412-0196
            doc_count += 1
            text = doc_chunk[docname]

            # assigning ids to docnames and vice versa
            if docname not in docid_vocab["ntoi"]:
                new_idx = len(docid_vocab["ntoi"]) + 1 #appending 1 to allot unique id to each docname
                docid_vocab["ntoi"][docname] = new_idx
                docid_vocab["iton"][new_idx] = docname

            tokens, vocab = tokenize_build(text, args.do_stem, stopwords, vocab, docid_vocab["ntoi"][docname])
            doc_len[docname] = len(tokens)
            for term, docid, position in tokens:
                if term not in partial_inverted_index:
                    partial_inverted_index[term] = {"ts": {docid: [position]}}
                    partial_inverted_index[term]["fs"] = {"ttf": 1}
                    sum_ttf += 1
                else: # if term id is in inverted_index but docid is not there
                    if docid not in partial_inverted_index[term]["ts"]:
                        partial_inverted_index[term]["ts"][docid] = [position]
                        partial_inverted_index[term]["fs"]["ttf"] += 1
                        sum_ttf += 1
                    else:
                        partial_inverted_index[term]["ts"][docid].append(position)
                        partial_inverted_index[term]["fs"]["ttf"] += 1
                        sum_ttf += 1

        write_to_file(partial_inverted_index, index, args.outf, args.compress)
    print("Vocab : %d" % len(vocab))
    print("doc count %d:" % doc_count)
    print("sum_ttf %d" % sum_ttf)
    vocab = {"token_vocab": vocab, "doc_vocab": docid_vocab}
    vocab["sum_ttf"] = sum_ttf
    vocab["doc_count"] = doc_count
    vocab["doc_len"] = doc_len
    save_vocab(vocab, args.outf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--dir_path", type=str, default="../data/AP_DATA/ap89_collection/", help="")
    parser.add_argument("--do_stem", action='store_true', help="")
    parser.add_argument("--compress", action='store_true', help="")
    parser.add_argument("--stopfile", type=str, default="data/stoplist.txt", help="")
    parser.add_argument("--outf", type=str, default="custom", help="")
    args = parser.parse_args()
    print(args)
    main(args)
