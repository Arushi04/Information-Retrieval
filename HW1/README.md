# Information-Retrieval

### Installation steps for Docker for MAC and Elastic Search & Kibana using Docker:

1. Install Docker Desktop for Mac OS from https://docs.docker.com/install/
2. Verify the installation by checking the version and running hello-world from Docker hub.
docker --version
docker run hello-world

3. Now copy the 3 yml files to your local from the path 'Information-Retrieval/data':
a) docker-compose.yml
b) elasticsearch.yml
c) kibana.yml

4. Run the below command in the same location where the above files are kept to get Kibana and elastic search up and running.
docker-compose up

5. In the browser, navigate to localhost:9200 to confirm Elasticsearch connection and localhost:5601 to confirm Kibana connection.

-------------------------------------------------------------------------

Commands to run the project:

1. python indexing.py : creates an index of the data on Elastic search with the given settings
2. python precompute_Stats.py : creates a dictionary with all the required details of term and field statistics and dumps it as pickle file.
3. python queryImplementation.py \
--index_name hw1_dataset \
--queryfile queryfile1.txt \
--output default/     
Implementation of all the vector and language models against the indexed data on ES.

4. EC1 :  Adding significant terms with high tf for better precision.    
python run_ec1_ec2.py \
--outf queryfile_ec1.txt \
--method ec1 --cutoff_per_query 3   
EC2 : Adding synonyms of the stemmed query terms to the query.  
python run_ec1_ec2.py \
--outf queryfile_ec2.txt 
--method ec2 \
--cutoff_per_query 1 \
--query_fname queryfile.txt