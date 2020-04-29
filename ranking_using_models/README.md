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

### Results of each model:

|**Model Names   | Average Precision   | Precision at 10  | Precision at 30**    |  
|--------------- | ------------------  | ----------------  | ------------------  |
|ES Built in     |	     0.2063	    |     0.3720	    |       0.3347.       |
|Okapi tf	       |       0.0736	     |     0.3800	     |     0.3120.       |
|Tf-idf	       |       0.2256	       |   0.4240	      |    0.3547.      |
|Okapi BM25	  |       0.2046	      |    0.4000	     |     0.3507.      |
|Laplace Smoothing |	0.0522	        |  0.3960	     |     0.3027.      |
|Jelinek Mercer Smoothing |	0.1325	   |  0.3520	     |     0.3107.      |


### Performance Enhancement:

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
