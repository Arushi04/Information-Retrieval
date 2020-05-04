Steps to run:

1. Run the preprocessing file to extract the data from email, get the clean text after removing stopwords, punctuations and upload to ES after splitting into training and testing.

python preprocessing.py \
--dirpath <path to emails folder> \
--labels <path to labels index file> \
--index <ES index name> \
--seed 4

2. Get the unigrams from the Elastic Search:

python getUnigrams.py \
--index <ES index name>

3. Build matrix and run the models:

python spamClassifier.py \
--index <ES index name> \
--labels <path to labels index file> \
--features <features file path> \
--cutoff <no of features u want to select from unigrams> \
--result <output path folder> \
--model <model types are : reg, logit(default), tree, nb(naive bayes)> \
--sparse (use only when u want to create sparse matrix)