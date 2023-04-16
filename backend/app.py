import json
import os
import re
import numpy as np
import math
from nltk.tokenize import TreebankWordTokenizer
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler

# ROOT_PATH for linking with all your files.
# Feel free to use a config.py or settings.py with a global export variable
os.environ["ROOT_PATH"] = os.path.abspath(os.path.join("..", os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
MYSQL_USER = "root"
MYSQL_USER_PASSWORD = "MayankRao16Cornell.edu"
# "MayankRao16Cornell.edu"
MYSQL_PORT = 3306
MYSQL_DATABASE = "coffeedb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE
)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(
    __name__,
    static_folder="static/react",
    template_folder="static/react",
    static_url_path="",
)
CORS(app)


# load data into a dict
def load_data():
    query_sql = f"""SELECT * from reviews"""
    data = mysql_engine.query_selector(query_sql)
    keys = ["id", "name", "roast", "100g_USD", "origin", "review"]
    return json.dumps([dict(zip(keys, i)) for i in data])


data = load_data()  # string of dictionaries
data_list = json.loads(data)  # convert to list of dicts


# Cosine Sim Algorithm
def tokenize(text):
    """Returns a list of words that make up the text.
    Params: {text: String}
    Returns: List
    """
    return re.findall("[a-z]+", text.lower())


def tokenize_reviews(coffee_data):
    """
    Returns a dictionary with all reviews and their tokenized words
    """
    tokens = set()
    review_dict = dict()
    for bean in coffee_data:
        name = bean["name"]
        review = bean["review"]
        t_review = tokenize(review)
        if name in review_dict:
            review_dict[name] += t_review
        else:
            review_dict[name] = t_review
    return review_dict


def build_inverted_index(review_dict):
    inverted_index = dict()  # dictionary with word: list of tuples
    doc_id = 0
    for bean, review in review_dict.items():  # go thru each dict
        # create a temp dict for count of words in tokenized_dict
        temp_dict = {}
        for token in review:
            temp_dict[token] = temp_dict.get(token, 0) + 1  # get count of each token

        # go thru every word in temp_dict
        for word, count in temp_dict.items():
            if word in inverted_index:
                inverted_index[word].append((doc_id, count))
            else:
                inverted_index[
                    word
                ] = list()  # initialize as list first idk if necessary
                inverted_index[word].append((doc_id, count))
        # move onto next doc
        doc_id += 1

        # now add counts to overall dictionary

    return inverted_index  # index w doc_id, count


def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    """Compute term IDF values from the inverted index."""

    idf_vals = dict()
    max_thresh = max_df_ratio * n_docs
    for term, docs in inv_idx.items():
        # print(type(docs))
        len_docs = len(docs)
        if len_docs <= max_thresh and len_docs >= 10:
            pre_log_idf = n_docs / (1 + len_docs)
            idf = math.log2(pre_log_idf)
            idf_vals[term] = idf
    return idf_vals


def compute_doc_norms(index, idf, n_docs):
    """Precompute the euclidean norm of each document.

    norms: np.array, size: n_docs
        norms[i] = the norm of document i.
    """

    norms = np.zeros(n_docs)
    for word in index:
        if word in idf:
            idf_weight = idf[word]
        else:
            idf_weight = 0  # prune to 0
        for doc in index[word]:
            tf_weight = doc[1]
            doc_id = doc[0]
            norms[doc_id] += (tf_weight * idf_weight) ** 2
    norms = np.sqrt(norms)
    # go thru all possible docs, find the word and its invertex index,
    # keep sum of product of tf number of times the word i appears in document j * idf[word]
    return norms


def accumulate_dot_scores(query_word_counts, index, idf):
    """Perform a term-at-a-time iteration to efficiently compute the numerator term of cosine similarity across multiple documents.

    Returns
    =======

    doc_scores: dict
        Dictionary mapping from doc ID to the final accumulated score for that doc
    """

    doc_scores = dict()

    for word, qf in query_word_counts.items():
        if word in index:
            documents = index[word]
            for doc in documents:
                doc_id, tf = doc[0], doc[1]

                if word not in idf:
                    idf_val = 0
                else:
                    idf_val = idf[word]

                acc = idf_val * qf * tf * idf_val
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = acc
                else:
                    doc_scores[doc_id] = doc_scores[doc_id] + acc
    return doc_scores


# takes in preferred country and output of indexsearch to recommend countries
def roast_search(results, data_list, roast):
    roast_output = list()
    for cossim_val, bean_id in results:
        bean_info = data_list[bean_id]
        if bean_info["roast"] == roast:
            roast_output.append((cossim_val, bean_id))
    return roast_output


def index_search(
    query,
    roast_value,
    index,
    idf,
    doc_norms,
    tokenizer,
    score_func=accumulate_dot_scores,
):
    """Search the collection of documents for the given query

    Returns
    =======

    results, list of tuples (score, doc_id)
        Sorted list of results such that the first element has
        the highest score, and `doc_id` points to the document
        with the highest score.

    """
    query = query.lower()
    query_tokens = tokenize(query)
    query_word_counts = dict()

    for word in query_tokens:
        query_word_counts[word] = query_word_counts.get(word, 0) + 1
    results = list()
    doc_scores = score_func(query_word_counts, index, idf)
    # q_norms
    q_norm = 0
    for term, freq in query_word_counts.items():
        if term in idf:
            idf_weight = idf[term]
        else:
            idf_weight = 0  # prune to 0
        q_norm += (freq * idf_weight) ** 2
    q_norm = math.sqrt(q_norm)

    for doc_id, doc_score in doc_scores.items():
        cossim_val = doc_score / (doc_norms[doc_id] * q_norm)
        results.append((cossim_val, doc_id))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    roast_results = roast_search(
        results, data_list, roast_value
    )  # top roast results (may not be anything)

    print(roast_results)
    difference = set(results) - set(roast_results)
    final_results = roast_results + list(difference)
    print(final_results)
    return final_results[0:10]  # return first top ten similar


review_dict = tokenize_reviews(data_list)
inv_idx = build_inverted_index(review_dict)
idf = compute_idf(inv_idx, len(review_dict), min_df=10, max_df_ratio=0.1)

inv_idx = {
    key: val for key, val in inv_idx.items() if key in idf
}  # prune the terms left out by idf
bean_doc_norms = compute_doc_norms(inv_idx, idf, len(review_dict))


# directly serch for roast values
def filter_search(roast):
    query_sql = f"""SELECT * from reviews WHERE roast =={roast}"""
    data = mysql_engine.query_selector(query_sql)
    return list(data)


def get_top_10_rec(
    query,
    roast_value,
    inv_idx=inv_idx,
    idf=idf,
    bean_doc_norms=bean_doc_norms,
    tokenize=tokenize,
):
    output = index_search(
        query, roast_value, inv_idx, idf, bean_doc_norms, tokenize
    )  # score, doc id
    rec_beans = (
        list()
    )  # list of tuples of top 10 recommended beans s.t. (name, cossim_score)
    for score, bean_id in output:
        bean_info = data_list[bean_id]
        rec_beans.append({"bean_info": bean_info, "score": score})
    return rec_beans


# rel_beans should be top 10 most similar cbeans & reviews & what frontend displays
# end of new code


# renders home page
@app.route("/")
def home():
    return render_template("index.html")


# return search recommendations
@app.route("/beans")
def beans_search():
    flavor_prof = request.args.get("flavor_prof")
    roast_value = request.args.get("roast_value")
    return get_top_10_rec(flavor_prof, roast_value)


app.run(debug=True)
