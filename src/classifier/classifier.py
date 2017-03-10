from __future__ import division
import re
import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from string import punctuation

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class SnowballStemTokenizer(object):
    def __init__(self):
        self.ss = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.ss.stem(t) for t in word_tokenize(doc)]

class PorterStemTokenizer(object):
    def __init__(self):
        self.ps = PorterStemmer()
    def __call__(self, doc):
        return [self.ps.stem(t) for t in word_tokenize(doc)]


def csv_to_df(path):
    '''
    Input: path.

    Save pandas dataframe to csv file.
    '''
    return pd.read_csv(path)

def count_vectorizer(X, max_features=None, tokenizer=None):
    '''
    Fits and transforms data into a count vector (matrix of token counts).
    '''
    cv = CountVectorizer(max_features=max_features,
                         tokenizer=tokenizer,
                         stop_words='english',
                         max_df=0.7)
    fitted_count_vector = cv.fit(X)
    count_vector = cv.transform(X)
    return count_vector, fitted_count_vector

def fit_tfidf(count_vector):
    tfidf = TfidfTransformer(use_idf=False)
    tfidf_vector = tfidf.fit(count_vector)
    return tfidf_vector

def transform_tfidf(tfidf_vector, count_vector):
    return tfidf_vector.transform(count_vector)

def nmf(X, n_components=None):
    model = NMF(n_components)
    W = model.fit_transform(X)
    components = model.components_
    return W, components

def lda(X, n_topics=None):
    model = LatentDirichletAllocation(n_topics)
    X_new = model.fit_transform(X)
    pass

def open_pickled_model(path):
    with open(path) as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    # Load fitted model
    model = open_pickled_model('gb_model.pkl')
    fitted_count_vector = open_pickled_model('count_vector.pkl')

    # Load unlabeled data
    df = csv_to_df('nyt_data.csv')
    df = df[['Article', 'Date', 'Title', 'URL', 'y']]
    df = df[df.y.isin([1,0]) == False]
    X = df[['Article','Date', 'Title', 'URL']]
    X.reset_index(inplace=True)
    y = df['y']

    # Transform data using fitted count vector
    count_vector = fitted_count_vector.transform(X['Article'])
    tfidf_vector_fit = fit_tfidf(count_vector)
    tfidf_vector_transform = transform_tfidf(tfidf_vector_fit, count_vector)

    # Predict on gradient boosting model
    y_pred = model.predict(tfidf_vector_transform.toarray())

    # Get indices of articles, which were classified as 1
    topic_indices = np.where(y_pred==1)[0]

    # Select corresponding articles from dataframe
    topic_df = X.iloc[topic_indices]

    # Save articles to csv file
    pd.to_csv('topic_articles.csv')
