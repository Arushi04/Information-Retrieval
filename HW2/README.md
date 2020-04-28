# Information-Retrieval

#### Steps to run the project:

1. Creating index on local. The below commands dumps partial inverted index files and partial catalog files.

a) Creating stemmed index :   
python createIndexing.py --do_stem --outf output/stemmed/

b) Creating unstemmed index:   
python createIndexing.py --outf output/unstemmed/

c) Creating compressed stemmed index:    
python createIndexing.py --do_stem --compress --outf output/stem_compressed/


2. Merging : This step merges all the partial indexes to create a final merged inverted index.

a) Uncompressed stemmed:   
python merging.py \
--dirpath output/stemmed/ 


b) Compressed stemmed :    
python merging.py \
--compress \
--dirpath output/stem_compressed/

3.        
a)Uncompressed stemmed :     
python runModels.py \
-qf data/queryfile1.txt \
-stem \
-i output/stemmed/combined_inverted_index_84.txt \
-vocab output/stemmed/vocab.pickle \
-c output/stemmed/combined_catalog_84.txt \
--ptype min \
--p_alpha .1 \
-o output/models/stemmed/     

 b) Compressed stemmed :     
python runModels.py \
-qf data/queryfile1.txt \
-stem \
--compress \
-i output/stemmed_compressed/combined_inverted_index_84.txt \
-vocab output/stemmed_compressed/vocab.pickle \
-c output/stemmed_compressed/combined_catalog_84.txt \
--ptype min \
--p_alpha .1 \
-o output/models/stemmed_compressed/ 


Running trec file for precision evaluation of all models:
./trec.pl -q data/qrel.txt output/models/stemmed/tf-idf.out


