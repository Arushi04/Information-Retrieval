As a first step, we have cleaned the data and indexed it on elastic search. Now, we will be matching the queries against the indexed dataset on ES, rank the documents for each query and record the performance for each vector and language model.

**_Models used are :_**      
**a) Vector Models :** ES-built-in(default), Okapi-tf, tf-idf, Okapi-BM25.  
**b) Language Models :** Unigram LM with Laplace smoothing, Unigram LM with Jelinek-Mercer smoothing.


### Files to run:   

1. *python precompute_Stats.py* : creates a dictionary with all the required details of term and field statistics and dumps it as pickle file. We have precomputed because it takes some time to fetch all the relevant data from ES.

2. *python queryImplementation.py*
*--index_name ap_dataset*
*--queryfile queryfile.txt*
*--output default/*.  
This file implements all the vector and language models to rank the documents indexed on elastic search against user queries and write the results to separate files.


## Performance Enhancement:

1. After seeing the performance above, we would be now adding terms with high term frequency to get better precision and highly relevant documents.    
      *python run_ec1_ec2.py*
      *--outf queryfile_ec1.txt*
      *--method ec1 --cutoff_per_query 3*

2. To increase the precision, we will now try with adding synonymns of the tsemmed query terms to the query.     
      *python run_ec1_ec2.py*
      *--outf queryfile_ec2.txt *
      *--method ec2*
      *--cutoff_per_query 1*
      *--query_fname queryfile.txt*
