# Information-Retrieval-Project
**About our project:**
Our project purpose is to build a search engine which is build of the corpus out of the whole English Wikipedia.

Now we will explain about the different classes, files and functions of our project:

**Inverted_index_title_gcp**:
This file is based on the 'inverted_index_gcp' file from the homework. It includes the inverted index class, and the classes that write and reads it the inverted index class include files such as: document frequency, term total amount, document lengths, posting locs dictionary.

**Inverted_index_body_gcp:**
This file is based on the 'inverted_index_gcp' file from the homework. It includes the inverted index class, and the classes that write and reads it the inverted index class include files such as: document frequency, term total amount, document lengths, posting locs dictionary.

**Inverted_index_anchor_gcp:**
This file is based on the 'inverted_index_gcp' file from the homework. It includes the inverted index class, and the classes that write and reads it the inverted index class include files such as: document frequency, term total amount, document lengths, posting locs dictionary.

**Search_frontend:**
This file includes all of the different searching methods which I will describe now:
At the beginning of the code we import all of the required packages, and defining the global variables:
1.	english_stopwords (nltk)
2.	corpus_stopwords (given at homework)
3.	all_stopwords (1+2)
4.	RE_WORD (given at homework)
5.	TEXT_INDEX 
6.	INDEX_TITLE
7.	page_ranks
8.	INDEX_ANCHOR
9.	page_view
10.	titles (dictionary of {doc_id: title, …})
11.	cosine_sim (dictionary of {query: {doc_id: cossim_score, …}, …})
12.	idf (dictionary of {token: idf_score, …})
13.	tf (dictionary of {token: tf_score,…})
14.	TUPLE_SIZE
15.	TF_MASK


**read_posting_list_title:**
reading the posting list of the title index, function from homework.

**read_posting_list_body:**
reading the posting list of the body index, function from homework.

**read_posting_list_anchor:**
reading the posting list of the anchor index, function from homework.

Help functions:

**tfidf**:
calculating the tf score and the idf score for each query with all of the documents that it appears on, according to it's posting list. Filling the idf dictionary and the tf dictionary.

**cossim**:
alculating the tfidf score for each query with all of the documents that it appears on, according to it's posting list. Then calculates the cosine similarity score and fills the cosine_sim dictionary.

**search_query:**
Returning the cosine similarity of specific query from to the cosine_sim dictionary.

**simple_tokenize:**
gets text and tokenize it's terms according to the tokenize function from homework.

**tokenize_search**:
gets text and tokenize it's terms according to the tokenize function from homework. We move plural forms to singular, removing '-', '_', and removing ''s' in ends of words.

**expand_fix:**
this function reduce the query's words using nltk package, we choose specific forms of words (verbs, nouns…) that we want to use for the query and remove the others. 

**binary_ranking:**
this function gets tokens of query and index and calculating the amount of tokens that appear in each document, and returning a list of doc id's in descending order according to the amount of tokens that appear in this document's title.

**binary_ranking_anchor:**
this function gets tokens of query and index and calculating the amount of tokens that appear in each document, and returning a list of doc id's in descending order according to the amount of tokens that appear in this document's anchor_text.

**page_rank:**
returning list of the page ranks for given doc id's

**page_view:**
returning list of the page views for given doc id's


main functions:

**search:** 
this is the most interesting function!
Here we making the query expansion and reduction, according to tokenize_search and expand_fix. Then we are doing searching according to document's titles. Then we choose the top 100 results and sorting them by their page rank. In this way we get to average MAP@40 of 0.44!!  

**search_body:**
using the cosine similarity score between documents and queries it returns the top 100 most similar documents for a given query. Using the tfidf, cossim and search_query. 

**search_title:**
using the binary ranking method it returns a list of doc id's and titles in descending order according to the amount of tokens that appear in this document's title.

**search_anchor:**
using the binary ranking method it returns a list of doc id's and titles in descending order according to the amount of tokens that appear in this document's anchor text.

**get_pagerank:**
returns the page ranks for a given list of doc ids.

**get_pageview**
returns the page views for a given list of doc ids.



*We will mention that the main difference between the different inverted_index_gcp files are the paths that the files are saved on. 
