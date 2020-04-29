# Information-Retrieval

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
