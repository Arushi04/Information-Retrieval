import argparse
import os
import io
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *
import re
import pickle
from itertools import islice

'''
Running the file:
with stem : python createIndexing.py --do_stem --outf output/custom_stem
without stem : python createIndexing.py --outf output/custom_nostem
'''

def save_data(data,vocab, fname):
    print("Writing inverted index and vocab to pickle file %s.[vocab/index]" % fname)
    pickle.dump(data, open("%s.index" % fname, "wb"))
    pickle.dump(vocab, open("%s.vocab" % fname, "wb"))

def my_stemmer(word):
    '''
        return stem of the word
    '''
    stemmer = PorterStemmer()
    stem = stemmer.stem(word)
    #stem = SnowballStemmer("english").stem(word)
    return stem


def my_tokenize(text, do_stem, stopwords, vocab, docid):
    '''
        This function tokenizes the code and do stemming according to do_stem status(True/False)
        U.S -> u . s .
        (1, 20, 1), (2, 20, 2), (3, 20, 3), (4, 20, 4), (1, 20, 5), (2, 20, 6), (5, 20, 7)
    '''
    token_list = []
    tokens = re.findall(r"([0-9\.0-9]+|[a-z]+|[\!\.,;:`-]+|[\w'\w]+)", text.lower())
    position = 0
    for token in tokens:
        if token in stopwords:
            continue
        if do_stem:
            token = my_stemmer(token)

        if token not in vocab:
            token_id = len(vocab) + 1
            vocab[token] = token_id
        else:
            token_id = vocab[token]

        position += 1
        token_list.append([token_id, docid, position])
    #print("Original Text : %s" % text)
    #print(token_list)
    return token_list, vocab


def read_data(path):
    '''
        reads all files in the directory in 'path'. doc_data is a dictionary with docid as key and text as val
    '''
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

def read_stopwords(fname):
    stopwords = set()
    with io.open(fname, 'r', encoding='ISO-8859-1') as f:
        for word in f:
            stopwords.add(word.strip())
    return stopwords


def divide_chunks(data, SIZE=1000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}


def main(args):
    doc_data = read_data(args.dir_path)
    stopwords = read_stopwords(args.stopfile)
    vocab = {}   #
    docid_vocab = {}   #stored docnames as keys and docids(1,2,3) as values
    inverted_index = {}
    '''Divide the data in chunks of 1000'''
    data = divide_chunks(doc_data)
    for term in data:  #looping each term in the 1000 batch
        for docname in doc_data:   #docname : AP890412-0196
            text = doc_data[docname]

            if docname not in docid_vocab:
                docid_vocab[docname] = len(docid_vocab) + 1    #appending 1 to allot id to each docname

            tokens, vocab = my_tokenize(text, args.do_stem, stopwords, vocab, docid_vocab[docname])
            #print("vocab size %d" % len(vocab))

            for termid, docid, position in tokens:
                if termid not in inverted_index:
                    inverted_index[termid] = {"ts" : {docid : [1 , [position]]}}
                    inverted_index[termid]["fs"] = {"df" : 1, "ttf" : 1}
                else: # if term id is in inverted_index but docid is not there
                    if docid not in inverted_index[termid]["ts"]:
                        inverted_index[termid]["ts"][docid] = [1, [position]]
                        inverted_index[termid]["fs"]["df"]+=1
                        inverted_index[termid]["fs"]["ttf"] += 1
                    else:
                        inverted_index[termid]["ts"][docid][0]+=1
                        inverted_index[termid]["ts"][docid][1].append(position)
                        inverted_index[termid]["fs"]["ttf"] += 1

            '''
                create indexing here
                {term1:[docid1,count,[positions]}{
                {
                    'term1' : {
                                 term_statistics : {
                                                    docid1 : [
                                                            tf , [pos1,pos22,pos34]
                                                            ],
                                                    docid2 : [
                                                            tf , [pos10,pos221,pos341]
                                                            ],
                                                    }
                                field_statistics:{
                                                    DF : in how many docs does a term appear
                                                    TTF : total no of the term in the corpus
                                }
                    'term2' : { docid3 : [tf,[pos8,pos220,pos134]], docid4 : [tf,[pos11,pos122,pos304]]}
                }
                DF: no of docs that contains the term -> len(inverted_index['term']
                CF(TTF): 
                list : [token_id, docid, position])
            '''
    print("Vocab : %d" % len(vocab))
    save_data(inverted_index, vocab, args.outf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index_name", type=str, default="hw1_dataset1", help="")
    parser.add_argument("--dir_path", type=str, default="/Users/arushi/Documents/Spring2020/CS6200/Assignments/AP_DATA/ap89_collection", help="")
    parser.add_argument("--do_stem", action='store_true', help="")
    parser.add_argument("--stopfile", type=str, default="stoplist.txt", help="")
    parser.add_argument("--outf", type=str, default="custom", help="")
    args = parser.parse_args()

    main(args)