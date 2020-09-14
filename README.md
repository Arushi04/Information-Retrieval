# Information-Retrieval

### Description :
This project covers a lot of concepts of Information Retrieval and text processing. I constructed elastic search index, used vector space models and
language models to rank the documents, developed web scraper using Beautiful soup and BFS algorithm.  Scraped and ranked documents using page ranking
algorithm and built my own trec eval file that assess model's performance.

Projects are done in the order:

1. [elastic_search](/elastic_search) : Cleaned the dataset which consists around 85000 documents and indexed document numbers and their corresponding text on the Elastic Search.

2. [ranking_using_models](/ranking_using_models) : Ranked the documents indexed in elastic search using vector and language models and compared the model performances.

3. [self_indexing](/self_indexing) : Created an index (similar to Elastic Search), indexed the data and compared how the models rank those indexed data and how  performance differs when compared to that of Elastic Search.

4. [web_crawling](/web_crawling) : Wrote a web crawler which takes few seed urls as input, do the crawling of the most relevant pages with politeness using Breadth First Search and stores the data on Elastic Search server. This project also consists of a merging code on ES server which helped me to merge my crawled data with that of my teammates.

5. [pagerank](/pagerank) : Created the inlinks and outlinks graph from the crawled data and computed their pageranks. Calculated the top hubs and authorities for the top ranking pages using HITS algorithm.

6. [creating_trecEval](/creating_trecEval) : Trec eval is the script that is used to assess the model performances. Wrote this script in python and it returns the precision, accuracy and recall for each query.


### Installation steps for Docker for MAC and Elastic Search & Kibana using Docker:

1. Install Docker Desktop for Mac OS from https://docs.docker.com/install/ 
2. Verify the installation by checking the version and running hello-world from Docker hub.
*docker --version*
*docker run hello-world*

3. Now copy the config folder that contains below files:
a) docker-compose.yml
b) elasticsearch.yml
c) kibana.yml

4. Run the below command in the same location where the above files are kept to get Kibana and elastic search up and running.
*docker-compose up*

5. Navigate to localhost:9200 to confirm Elasticsearch connection and localhost:5601 to confirm Kibana connection.


