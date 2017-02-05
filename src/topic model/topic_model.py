from __future__ import division
import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from string import punctuation
from textblob import TextBlob

def csv_to_df(path):
    ''' Loads csv into pandas dataframe '''
    return pd.read_csv(path)

def count_vectorizer(X, max_features=None, tokenizer=None):
    '''
    Generates a count vector.
    Parameters are maximum features of vector and tokenizer class.
    Outputs a transformed count vector and a fitted count vector.
    '''
    cv = CountVectorizer(max_features=max_features, tokenizer=tokenizer, stop_words='english', max_df=0.7)
    fitted_count_vector = cv.fit(X)
    count_vector = cv.transform(X)
    return count_vector, fitted_count_vector

def fit_tfidf(count_vector):
    '''
    Fits a term frequency matrix on a count vector.
    '''
    tfidf = TfidfTransformer(use_idf=False)
    tfidf_vector = tfidf.fit(count_vector)
    return tfidf_vector

def transform_tfidf(tfidf_vector, count_vector):
    '''
    Transforms prefitted count vector.
    '''
    return tfidf_vector.transform(count_vector)

def final_tf_vector(X, max_features=None, tokenizer=None):
    '''
    In lieu of a TfidfVectorizer.
    Generates a count vector and, fits and transforms a term frequency matrix.
    Outputs a transformed tf vector and feature words.
    '''
    count_vector, fitted_count_vector = count_vectorizer(X, max_features, tokenizer)
    tfidf_vector_fit = fit_tfidf(count_vector)
    tfidf_vector_transform = transform_tfidf(tfidf_vector_fit, count_vector).toarray()
    feature_words = fitted_count_vector.get_feature_names()
    return tfidf_vector_transform, feature_words

def remove_punctuation(df):
    '''
    Removes punctuation from dataframe.
    '''
    punct = punctuation
    pattern = r"[{}]".format(punct) # create the pattern
    X = [re.sub(pattern, "", article) for article in df['Article']]
    return X

def nmf(X, n_components=None):
    '''
    Non Negative Matrix Factorization.
    Outputs the weights (W) matrix and the components.
    '''
    model = NMF(n_components)
    W = model.fit_transform(X)
    components = model.components_
    return W, components

def cluster_counts(W):
    '''
    Selects the topic for each article.
    For each article uses the topic, which describes it the most.
    Prints the number of articles in each topic and outputs the index of the topic for each article.
    '''
    topic_indices = []
    vector = pd.DataFrame(W)
    for i in xrange(len(vector)):
        topic_indices.append(vector.iloc[i,:].argmax())
    print pd.Series(topic_indices).value_counts()
    return topic_indices

def top_topic_words(feature_words, components, num_top_words):
    '''
    Prints the topic number and the top 10 words associated with each topic.
    '''
    topic_words = []
    for topic in components:
        word_idx = np.argsort(topic)[::-1][0:num_top_words]
        topic_words.append([feature_words[i] for i in word_idx])

    for t in range(len(topic_words)):
        print("Topic {}: {}".format(t, ' '.join(topic_words[t])))

def lda(X, n_topics=None):
    model = LatentDirichletAllocation(n_topics)
    X_new = model.fit_transform(X)
    components = model.components_
    return X_new, components

if __name__ == '__main__':
    # Load data
    df = csv_to_df('topic_articles.csv')

    # Remove punctuation
    X = remove_punctuation(df)

    # Produce TF vector and feature words matrix
    tfidf_vector_transform, feature_words = final_tf_vector(X=X, max_features=10000, tokenizer=LemmaTokenizer())

    # Produce W matrix and components
    W, components = nmf(tfidf_vector_transform, n_components=5)

    # Index of topic for each article
    topic_indices = cluster_counts(W)

    # Top words for  each topic
    top_topic_words(components, num_top_words=15)
