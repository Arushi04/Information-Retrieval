### Information-Retrieval

Projects are done in the order:

1. **elastic_search** : Here I have explored elastic search. I have cleaned the dataset which consists of around 85000 documents and their text and indexed it on the elastic search.

2. **ranking_using_models** : I have ranked the documents indexed in elastic search using vector and language models and have compared the model performances.

3. **self_indexing** : Here, I have created an index on my system(similar to elastic search index), indexed the data and compared how the models rank those indexed data and how close we are in performance if compared to that of elastic search.

4. **web_crawling** : Wrote a web crawler which takes few seed urls as input and do the crawling of the most relevant pages with politeness and stores the data on elastic search server. This project also consists of a merging code which helped me to merge my crawled data with that of my teammates.

5. **pagerank** : Created the inlinks and outlinks graph from the crawled data and computed their pageranks. HAvecalculated the top hubs and authorities for the top ranking pages using HITS algorithm.

6. **creating_trecEval** : Trec eval is the script that I have used to assess the model performances. I have written this script in python and it returns the precision, accuracy and recall for each query.

7. **spam_classification** : Here I have worked on spam classification of emails using Logistic Regression, Decision trees and Multinomial Naive Bayes algorithms. This project includes reading the email format of data and cleaning the data. For cleaning, I have used spacy library and have indexed the cleaned data on elastic search for further classification. 


