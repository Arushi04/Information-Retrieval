Steps to run:

This has been run on the indexed AP_dataset on ES. The program gets the data from ES and creates a sparse matrix for each query and run LDA on it. 
The code is only for Part A of the assignment.

python topicModelling.py \
--index <ES index name> \
--rfile <path of bm25 file> \
--qrel <path to qrel file> \
--queries <path to query file>
