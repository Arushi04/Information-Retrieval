import pickle
import tqdm
import zlib
import argparse
from custom_util import get_posting, format_posting, encode_this

'''
Running the file:
with stem : python merging.py --dirpath output/stem_updated/
without stem : python merging.py --dirpath output/unstemmed/
with compression : python merging.py --compress --dirpath output/stemmed_compressed/
'''
parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument("--compress", action='store_true', help="")
parser.add_argument("--dirpath", type=str, default="output/stemmed/", help="")
args = parser.parse_args()


def combine_these_indices(current_cat, last_combined_cat, current_invf, last_combined_invf, new_combined_invf_fname, compress):
    '''
        last_combined_cat and last_combined_invf can be NONE
    '''
    fo = open(new_combined_invf_fname, "wb")
    catalog = {}
    last_offset = 0

    if last_combined_cat:
        # Check if this is None
        for term in last_combined_cat:
            position = last_combined_cat[term]
            total_ttf, total_postings = get_posting(term, last_combined_invf, position, compress)

            if term in current_cat:
                position = current_cat[term]
                ttf_s, postings_s = get_posting(term, current_invf, position, compress)
                total_ttf += ttf_s
                total_postings += postings_s

            line = format_posting(term, total_ttf, total_postings, compress)
            catalog[term] = [last_offset, len(line)]
            last_offset += len(line)
            fo.write(encode_this(line))

    for term in current_cat:
        if term in catalog:
            continue
        position = current_cat[term]
        ttf, postings = get_posting(term, current_invf, position, compress)
        line = format_posting(term, ttf, postings, compress)

        catalog[term] = [last_offset, len(line)]
        last_offset += len(line)
        fo.write(encode_this(line))

    fo.close()
    return catalog


for idx in tqdm.tqdm(range(85)):

    current_cat_fname = args.dirpath + "partial_catalog_%d.txt" % idx
    current_invf_fname = args.dirpath + "partial_inverted_index_%d.txt" % idx
    last_combined_cat_fname = args.dirpath + "combined_catalog_%d.txt" % (idx - 1)
    last_combined_invf_fname = args.dirpath + "combined_inverted_index_%d.txt" % (idx - 1)
    new_combined_cat_fname = args.dirpath + "combined_catalog_%d.txt" % idx
    new_combined_invf_fname = args.dirpath + "combined_inverted_index_%d.txt" % idx

    if idx == 0:
        # first time combined_xxx_0 doesn't exist
        last_combined_cat_fname = None
        last_combined_invf_fname = None
        last_combined_cat = None
        last_combined_invf = None
    else:
        last_combined_cat = pickle.load(open(last_combined_cat_fname, "rb"))
        last_combined_invf = open(last_combined_invf_fname,"rb")

    current_cat = pickle.load(open(current_cat_fname, "rb"))
    current_invf = open(current_invf_fname,"rb")

    combined_cat = combine_these_indices(current_cat, last_combined_cat,
                                         current_invf, last_combined_invf,
                                         new_combined_invf_fname, args.compress)

    # print(len(combined_cat))
    pickle.dump(combined_cat, open(new_combined_cat_fname, "wb"))

    current_invf.close()
    if last_combined_invf:
        last_combined_invf.close()