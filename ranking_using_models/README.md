# Document Ranking using IR models

The data has been cleaned and indexed on elastic search. Now, we will match the user queries against the indexed dataset on Elastic Search, rank the documents for each query and record the performance for each vector and language model.

**_Models used are :_**      
**a) Vector Models :** In VSM, we represent queries and documents as term vectors and compare similarity between them by doing dot product between 2 vectors.            
         ES-built-in(default), Okapi-tf, tf-idf, Okapi-BM25.  
**b) Language Models :** Language models tells us the probability of a sequence of words. It ranks the documents based on their probabilities to generate the query terms.       
Unigram LM with Laplace smoothing, Unigram LM with Jelinek-Mercer smoothing.


### Files to run:   

1. *python precompute_Stats.py* : creates a dictionary with all the required details of term and field statistics and dumps it as pickle file. We have precomputed because it takes some time to fetch all the relevant data from ES.

2. *python queryImplementation.py*
*--index_name ap_dataset*
*--queryfile queryfile.txt*
*--output default/*.  
This file implements all the vector and language models to rank the documents indexed on elastic search against user queries and write the results to separate files.

### Results of each model:

|**Model Names   | Average Precision   | Precision at 10  | Precision at 30**   |  
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
      
|**Model Names   | Average Precision   | Precision at 10  | Precision at 30**    |  
|--------------- | ------------------  | ----------------  | ------------------  |
|ES Built in	|0.3092	|0.4320	|0.3720|
|Okapi tf	|0.2550	|0.4360	|0.3320|
|Tf-idf	|0.3102	|0.4480	|0.3827|
|Okapi BM25	|0.3138	|0.4520	|0.3773|
|Laplace Smoothing	|0.2347	|0.4400	|0.3373|
|Jelinek Mercer Smoothing	|0.2933	|0.4000	|0.3667|




2. Now, we will try with adding synonymns of the stemmed query terms to the query to see if it improves the precision.    
      *python run_ec1_ec2.py*
      *--outf queryfile_ec2.txt *
      *--method ec2*
      *--cutoff_per_query 1*
      *--query_fname queryfile.txt*
      
      
|**Model Names   | Average Precision   | Precision at 10  | Precision at 30**    |  
|--------------- | ------------------  | ----------------  | ------------------  |
|ES Built in	|0.3077	|0.4360	|0.3587 |
|Okapi tf	|0.2716	|0.4560	|0.3573. |
|Tf-idf	|0.3098	|0.4520	|0.3840.  |
|Okapi BM25	|0.3160	|0.4560	|0.3813|
|Laplace Smoothing	|0.2486	|0.4280	|0.3480|
|Jelinek Mercer Smoothing	|0.2930	|0.4000	|0.3667|
      
 
