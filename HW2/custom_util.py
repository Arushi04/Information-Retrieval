from nltk.stem.porter import *
import io
import string
import zlib
from itertools import islice

fname = "data/stoplist.txt"


def my_stemmer(word):
    stemmer = PorterStemmer()
    stem = stemmer.stem(word)
    #stem = SnowballStemmer("english").stem(word)
    return stem


def read_stopwords(fname):
    stopwords = set()
    with io.open(fname, 'r', encoding='ISO-8859-1') as f:
        for word in f:
            stopwords.add(word.strip())
    print("Total no of stopwords: %d" % len(stopwords))
    return stopwords


def my_tokenize(text, do_stem, stopwords=None):
    #tokens = re.findall(r"([0-9\.0-9]+|[a-z]+|[\!\.,;:`-]+|[\w'\w]+)", text.lower())
    tokens = re.findall(r"([\d+.\d+]+|[a-z]+|[\!\._,;:`-]|[\w'\w]+)", text.lower())
    #punctuations = list('''!()-[]{};:'"\,<>./?@#$%^&*_~''')
    punctuations = list(string.punctuation)+["''"]
    if stopwords:
        tokens = [token for token in tokens if token not in stopwords]
    if do_stem:
        tokens = [my_stemmer(token) for token in tokens]
    if punctuations:
        tokens = [token for token in tokens if token not in punctuations]
    return tokens


def get_posting(get_this_token, fp, pos, compress):
    fp.seek(pos[0], 0)
    content = fp.read(pos[1])
    content = decompress_line(content, compress)

    content = content.strip().split(" ")
    token, ttf, postings = content[0], int(content[1]), content[2:]

    assert len(token) >= 1
    assert token == get_this_token

    return ttf, postings

def encode_this(text):
    if isinstance(text, bytes):
        return text
    return text.encode('ISO-8859-1')

def decode_this(text):
    if isinstance(text, bytes):
        return text.decode('ISO-8859-1')
    return text


def decompress_line(line, do_compress):
    if do_compress:
        line = zlib.decompress(line).decode('ISO-8859-1').strip()
    if isinstance(line, bytes):
        line = decode_this(line)
    return line


def compress_line(line, do_compress):
    if isinstance(line, str):
        line = encode_this(line)
    if do_compress:
        line = zlib.compress(line)
    return line


def get_stats(pos, fp, token_to_get, compress):
    #print(pos, doc_id, fp, token_to_get)
    tf, df, ttf = 0, 0, 0
    fp.seek(pos[0], 0)
    content = fp.read(pos[1])
    content = decompress_line(content, compress)

    content = content.strip().split()
    token, ttf, postings_str = content[0], int(content[1]), content[2:]
    assert token == token_to_get

    postings = {}
    for idx in range(0, len(postings_str) - 1, 2):
        key = int(postings_str[idx])
        val = list(map(int, postings_str[idx + 1].split(",")))
        postings[key] = val
    postings["ttf"] = ttf
    return postings


def format_posting(term, ttf, posting, compress):
    posting = " ".join(posting)
    line = "%s %d %s\n" % (term, ttf, posting)
    if compress:
        line = compress_line(line, compress)
    return line


def divide_chunks(data, size=1000):
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}
