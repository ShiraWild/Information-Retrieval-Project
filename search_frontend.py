from flask import Flask, request, jsonify
import pickle
from collections import Counter, defaultdict
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from contextlib import closing
import inverted_index_title_gcp as title
import inverted_index_body_gcp as body
import inverted_index_anchor_gcp as anchor
import re

nltk.download('averaged_perceptron_tagger')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Global variables

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
with open('/home/maayana14/postings_body/ind_body.pkl', 'rb') as f:
    TEXT_INDEX = pickle.load(f)
with open('/home/maayana14/postings_gcp/ind_title.pkl', 'rb') as f:
    INDEX_TITLE = pickle.load(f)
with open('/home/maayana14/postings_gcp/page_rank.pkl', 'rb') as f:
    page_ranks = pickle.load(f)
with open('/home/maayana14/postings_anchor/ind_anchor.pkl', 'rb') as f:
    INDEX_ANCHOR = pickle.load(f)
with open('/home/maayana14/postings_gcp/pageviews-202108-user.pkl', 'rb') as f:
    pageview = pickle.load(f)
with open('/home/maayana14/postings_gcp/doctitles.pkl', 'rb') as f:
    titles_dd = pickle.load(f)
    titles = dict(titles_dd)

cosine_sim = defaultdict(float)
idf = defaultdict(float)
tf = defaultdict(float)

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


# Reading the posting list of title index

def read_posting_list_title(inverted, w):
    with closing(title.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


# Reading the posting list of body index

def read_posting_list_body(inverted, w):
    with closing(body.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


# Reading the posting list of anchor index

def read_posting_list_anchor(inverted, w):
    with closing(anchor.MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


# Calculating the tf and idf for each word and document in each query

def tfidf(query, index, tf, idf, n):
    tokens = simple_tokenize(query)
    for tok in tokens:
        try:
            pls = read_posting_list_body(index, tok)
            tf_values = defaultdict(int)
            df = len(pls)
            idf[tok] = np.log2((n + 1) / (df + 1))  # calculating idf
            for obj in pls:
                doc_id = obj[0]
                tf_val = obj[1]
                tf_values[doc_id] = tf_val / index.DL[doc_id]  # calculating tf
            tf[tok] = list(tf_values.items())
        except:
            continue


# Calculating the cosine similarity for each word and document in each query

def cossim(query, index, tf, idf):
    words_num = len(query.split())
    docs_lens = defaultdict(int)
    tfidf_scores = defaultdict(float)
    for term in idf:
        for obj in tf[term]:
            doc_id = obj[0]
            tf_val = obj[1]
            idf_val = idf[term]
            doc_len = index.DL[doc_id]
            tfidf_scores[doc_id] += tf_val * idf_val  # calculating tfidf
            docs_lens[doc_id] = doc_len
    for d_id in docs_lens:
        tfidf_scores[d_id] = tfidf_scores[d_id] / (docs_lens[d_id] * words_num)  # calculating cosine similarity
    # sorting by cosine similarity in descending order
    cosine_sim[query] = sorted(tfidf_scores.items(), key=lambda item: item[1], reverse=True)[:100]


# Returns cosine similarity for given query
def search_query(query, index):
    tfidf(query, index, tf, idf, len(index.DL))
    cossim(query, index, tf, idf)
    val = cosine_sim[query]  # cosine similarity score
    return val


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize_search(query)  # first, tokenize the query
    new_query = expand_fix(tokens)  # next, expanding and reducing the query
    bin = binary_ranking(new_query, INDEX_TITLE)[:100]  # calculating binary ranking by the titles
    pr = page_rank(bin, page_ranks)  # calculating page ranks for the relevant documents
    docs_pr = dict(zip(bin, pr))  # zipping to dictionary
    pre_res = sorted(docs_pr.items(), key=lambda item: item[1], reverse=True)  # sorting by page rank
    res = [(x[0], titles[x[0]]) for x in pre_res]  # adding the titles to the doc_id results
    # END SOLUTION
    return jsonify(res)


def tokenize_search(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    text = re.sub("[^a-zA-Z0-9\s\S]+", "", text)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    more_words = []
    # Here we're going to fix the query by transferring plural forms to singular,
    # and removing some characters like '-', '_'...
    for token in list_of_tokens:
        token = token.replace("'s", "")
        token = token.replace("-", " ")
        token = token.replace("_", " ")
        if token[-2:] == 'es':
            token = token[:-2] + 'e'
        elif token[-1:] == 's':
            token = token[:-1]
        more_words.append(token)
    return more_words


def simple_tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    # Just tokenizing like in the homework
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


def expand_fix(tokens):
    # extracting type (v/n..) for each word
    words_types = nltk.pos_tag(tokens)
    # choosing the relevant types
    types = ['NN', 'VB', 'NNS', 'VBN', 'JJ', 'VBP']
    final_words = []
    # saving only the relevant words
    for w, t in words_types:
        if t in types:
            final_words.append(w)
    return final_words


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    pre_res = search_query(query, TEXT_INDEX)
    res = [(x[0], titles[x[0]]) for x in pre_res]  # adding the titles to the doc_id results
    # END SOLUTION
    return jsonify(res)


# Binary ranking for title index

def binary_ranking(tokens, index):
    tokens_new = []
    words = index.df.keys()
    # removing question marks
    for token in tokens:
        new_token = token.replace("?", "")
        tokens_new.append(new_token)
    docs_terms_amount = {}  # amount of terms from the query in the document
    for term in tokens_new:
        if term in words:
            try:
                term_pls = read_posting_list_title(index, term)  # posting list for each term
                term_docs = [x[0] for x in term_pls]  # documents each term appears on
                for doc in term_docs:
                    if doc != 0:
                        if doc not in docs_terms_amount:
                            docs_terms_amount[doc] = 1
                        else:
                            docs_terms_amount[doc] += 1  # counting if term appears on doc
            except:
                continue
    # sorting in descending order by amount of terms from the query in the document
    p_res = sorted(docs_terms_amount.items(), key=lambda item: item[1], reverse=True)
    # Saving only the doc id's
    res = [x[0] for x in p_res]
    return res


# Binary ranking for anchor index

def binary_ranking_anchor(tokens, index):
    tokens_new = []
    words = index.df.keys()
    # removing question marks
    for token in tokens:
        new_token = token.replace("?", "")
        tokens_new.append(new_token)
    docs_terms_amount = {}  # amount of terms from the query in the document
    for term in tokens_new:
        if term in words:
            try:
                term_pls = read_posting_list_anchor(index, term)  # posting list for each term
                term_docs = [x[0] for x in term_pls]  # documents each term appears on
                for doc in term_docs:
                    if doc != 0:
                        if doc not in docs_terms_amount:
                            docs_terms_amount[doc] = 1
                        else:
                            docs_terms_amount[doc] += 1  # counting if term appears on doc
            except:
                continue
    # sorting in descending order by amount of terms from the query in the document
    p_res = sorted(docs_terms_amount.items(), key=lambda item: item[1], reverse=True)
    # Saving only the doc id's
    res = [x[0] for x in p_res]
    return res


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = simple_tokenize(query)
    pre_res = binary_ranking(tokens, INDEX_TITLE)[:100]  # selecting top 100 results of best binary ranking
    res = [(x, titles[x]) for x in pre_res]  # adding the titles to the doc_id results
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    tokens = simple_tokenize(query)  # tokenizing the query
    pre_res = binary_ranking_anchor(tokens, INDEX_ANCHOR)[:100]  # selecting top 100 results of best binary ranking
    res = [(x, titles[x]) for x in pre_res]  # adding the titles to the doc_id results
    # END SOLUTION
    return jsonify(res)


# Attaching page rank for each document

def page_rank(wiki_ids, page_ranks):
    pr_lst = []
    for wiki_id in wiki_ids:
        if wiki_id in page_ranks.keys():
            pr_lst.append(int(page_ranks.get(wiki_id)))
    return pr_lst


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = page_rank(wiki_ids, page_ranks)
    # END SOLUTION
    return jsonify(res)


# Attaching page views amount for each document

def page_view(ids, pickl):
    res = []
    for id in ids:
        if id in pickl.keys():
            res.append(pickl[id])
    return res


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = page_view(wiki_ids, pageview)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080)
