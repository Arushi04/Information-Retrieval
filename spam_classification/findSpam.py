import argparse
from collections import Counter
import pickle
import tqdm


def create_ngrams(text, vocab, ngram=3):
    words = text.split()
    for widx in range(len(words)):
        for n in range(1, ngram+1):
            if widx + n < len(words):
                word = " ".join(words[widx:widx + n])
                vocab[n][word] += 1
    return vocab


def read_spam_labels(fname):
    spam_label = []
    count = 0
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip().split("/")
            label = line[0].split(" ")[0]
            file = line[2]
            if label == 'spam':
                spam_label.append(file)
                count += 1
    print("total spam docs : ", count)
    return spam_label


def write_ngrams(vocab, fname):
    with open(fname, "w") as fout:
        for key in vocab:
                result = vocab[key].most_common()
                for item in result:
                    final_input = f"{item[0]} {key} {item[1]}\n"  # viagra 1 5
                    fout.write(final_input)


def main(args):
    with open(args.savepath, "rb") as handle:
        data = pickle.load(handle)
    spam_label = read_spam_labels(args.labels)
    vocab = {n: Counter() for n in range(1, args.ngram + 1)}  # {unigram : {word:freq}, bigram : {word:freq}, trigram : {word : freq}}
    for key in tqdm.tqdm(data):
        if key in spam_label:
            text = ' '.join(data[key])
            vocab = create_ngrams(text, vocab, args.ngram)

    #write_ngrams(vocab, args.spam_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("--index", type=str, default="spam_data", help="")
    parser.add_argument("--ngram", type=int, default=3, help="")
    parser.add_argument("--labels", type=str, default="data/trec/full/index", help="")
    parser.add_argument("--savepath", type=str, default="data/data.pickle", help="")
    parser.add_argument("--spam_vocab", type=str, default="data/spam_vocab.txt", help="")
    parser.add_argument("--features", type=str, default="data/spam_vocab.txt", help="")

    args = parser.parse_args()
    main(args)