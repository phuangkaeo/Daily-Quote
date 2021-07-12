# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
from sqlalchemy import sql
from sqlalchemy.sql.sqltypes import DateTime
import uvicorn
from fastapi import FastAPI 

from typing import List
import datetime as dt
import databases
import sqlalchemy
from sqlalchemy import create_engine
from pydantic import BaseModel

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

# load the sentiment model
with open(
    join(dirname(realpath(__file__)), "sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


# cleaning the data
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
        
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
        
    # Return a list of words
    return text

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)
    
    # perform prediction
    prediction = model.predict([cleaned_review])
    output = int(prediction[0])
    probas = model.predict_proba([cleaned_review])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {0: "Negative", 1: "Positive"}
    
    # show results
    result = {"prediction": sentiments[output], "Probability": output_probability}
    return result

## Create Database
DATABASE_URL = "sqlite:src/test.db"

database = databases.Database(DATABASE_URL)

metadata = sqlalchemy.MetaData()

notes = sqlalchemy.Table(
    "emotion",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("text", sqlalchemy.String(255)),
    sqlalchemy.Column("sentiment", sqlalchemy.String(50)),
    sqlalchemy.Column("score", sqlalchemy.Float),
    sqlalchemy.Column("user", sqlalchemy.String(50)),
    sqlalchemy.Column("date_created", sqlalchemy.DateTime, default=dt.datetime.utcnow)
)

# default
engine = create_engine('mysql://phuangkaeo:simplon59@localhost/sentiment')
# engine = sqlalchemy.create_engine(
#     DATABASE_URL, connect_args={"check_same_thread": False}
# )

metadata.create_all(engine)

class TextIn(BaseModel):
    text: str
    user: str
    # date_created: dt.datetime.utcnow

# class Text(BaseModel):
#     id: int
#     text: str
#     sentiment: str
#     score: float

# @app.get('/text')
# async def insert_text():
#     # query = notes.select()
#     engine.find(TextIn)
#     return await engine.find(TextIn)
@app.get("/text")
def get_text(get_text : TextIn):
    query = """ SELECT text
                FROM textemotion;
            """
    values = [(get_text.text_date,)] # %s is always replace by a tuple
    db_cursor.executemany(query,values)

# @app.post('/text')
# async def get_text(text: TextIn):
#     await engine.execute("INSERT INTO emotion VALUES ('text', 'user')")
#     return text
