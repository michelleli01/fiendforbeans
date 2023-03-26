import json
import os
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
MYSQL_PORT = 3306
MYSQL_DATABASE = "coffeedb"

mysql_engine = MySQLDatabaseHandler(
    MYSQL_USER, MYSQL_USER_PASSWORD, MYSQL_PORT, MYSQL_DATABASE
)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db()

app = Flask(__name__)
CORS(app)

def load_reviews():
    query_sql=f"""SELECT review from reviews"""
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([row["review"] for row in data])

reviews = load_reviews()

# Sample search, the LIKE operator in this case is hard-coded,
# but if you decide to use SQLAlchemy ORM framework,
# there's a much better and cleaner way to do this
# def sql_search(episode):
#    query_sql = f"""SELECT * FROM episodes WHERE LOWER( title ) LIKE '%%{episode.lower()}%%' limit 10"""
#    keys = ["id", "title", "descr"]
#    data = mysql_engine.query_selector(query_sql)
#    return json.dumps([dict(zip(keys, i)) for i in data])

# renders home page
@app.route("/")
def home():
    return render_template("base.html", title="sample html")


# return search recommendations
@app.route("/beans")
def beans_search():
    text = request.args.get("bean")
    return sql_search(text)


app.run(debug=True)
