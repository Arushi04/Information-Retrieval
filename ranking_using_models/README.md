### Commands to run the project:   

*python precompute_Stats.py* : creates a dictionary with all the required details of term and field statistics and dumps it as pickle file.   

*python queryImplementation.py*
*--index_name hw1_dataset*
*--queryfile queryfile1.txt*
*--output default/*.  
Above file implements the vector and language models to rank the documents indexed on elastic search against user queries.


**_Models used are :_** 
**a) Vector Models :** ES-built-in(default), Okapi-tf, tf-idf, Okapi-BM25.  
**b) Language Models :** Unigram LM with Laplace smoothing, Unigram LM with Jelinek-Mercer smoothing.



**EC1 :** Adding significant terms with high tf for better precision.   
*python run_ec1_ec2.py*
*--outf queryfile_ec1.txt*
*--method ec1 --cutoff_per_query 3*


**EC2 :** Adding synonyms of the stemmed query terms to the query.   
*python run_ec1_ec2.py*
*--outf queryfile_ec2.txt --method ec2*
*--cutoff_per_query 1*
*--query_fname queryfile.txt*
