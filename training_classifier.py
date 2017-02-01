from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


''' Tokenizer Classes '''
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



def tts(X, y):
    '''
    Train, test, split function.
    Input: X and y dataframes.
    Output: training and testing dataframes for both X and y.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    return X_train, X_test, y_train, y_test

def csv_to_df(path):
    '''
    Reads a csv file into a dataframe.
    '''
    return pd.read_csv(path)

def count_vectorizer(X, max_features=None, tokenizer=None):
    '''
    Fits and transforms a count vector on X data.
    Outputs both the fitted and the transformed count vector.
    Fitted count vector can be used in a tfidf transformer.
    '''
    cv = CountVectorizer(max_features=max_features, tokenizer=tokenizer, stop_words='english', max_df=0.7)
    fitted_count_vector = cv.fit(X)
    count_vector = cv.transform(X)
    return count_vector, fitted_count_vector

def fit_tfidf(count_vector):
    '''
    Transforms a count vector into a tf vector.
    TF: count vector normalized on legnth of docs.
    '''
    tfidf = TfidfTransformer(use_idf=False)
    tfidf_vector = tfidf.fit(count_vector)
    return tfidf_vector

def transform_tfidf(tfidf_vector, count_vector):
    '''
    Transforms a tf vector that's fitted on a count vector.
    '''
    return tfidf_vector.transform(count_vector)


def plot_roc_curve(y_test, y_proba, pos_label=1):
    '''
    Plot ROC Curve.
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, label=roc_auc)
    plt.title('ROC Curve', fontsize=40, weight='bold')
    plt.ylabel('True Positive Rate', size=15, weight='bold')
    plt.xlabel('False Positive Rate', size=15, weight='bold')

def standard_confusion_matrix(y_true, y_pred):
    '''
    Input: predicted y probabilities, true y labels.
    Output: true positives, false positives, false negatives, true negatives, and confusion matrix.
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum(y_true*y_pred)
    fp = np.sum(((y_pred-y_true) == 1).astype(int))
    fn = np.sum(((y_true-y_pred) == 1).astype(int))
    tn = np.sum((y_pred == y_true).astype(int)) - tp
    cm = np.array([[tp, fp], [fn, tn]])
    return tp, fp, fn, tn, cm


def pickle_file(model, path):
    '''
    Pickles a fitted model.
    '''
    with open(path, 'w') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    df = csv_to_df('train_data_sub.csv')
    df = df[['Article', 'Date', 'Title', 'URL', 'y']]
    df = df[df.y.isin([1,0])]
    X = df[['Article','Date']]
    y = df['y']

    ''' Train Test Split '''
    X_train, X_test, y_train, y_test = tts(X, y)

    ''' Training Gradient Boosting '''
    # Training Data
    count_vector_train, fitted_count_vector_train = count_vectorizer(X=X_train['Article'],
                        max_features=10000, tokenizer=LemmaTokenizer())
    tfidf_vector_fit_train = fit_tfidf(count_vector_train)
    tfidf_vector_transform_train = transform_tfidf(tfidf_vector_fit_train, count_vector_train)

    ''' Pickle fitted count vector '''
    pickle_file(fitted_count_vector_train, 'count_vector.pkl')

    ''' Testing Gradient Boosting '''
    # Testing Data
    count_vector_test = fitted_count_vector_train.transform(X_test['Article'])
    tfidf_vector_fit_test = fit_tfidf(count_vector_test)
    tfidf_vector_transform_test = transform_tfidf(tfidf_vector_fit_test, count_vector_test)

    # Gradient Boosting Classifier
    # Instantiate
    GB = GradientBoostingClassifier(n_estimators=100)
    # Fit
    GB.fit(tfidf_vector_transform_train, y_train)
    # Predict
    y_pred = GB.predict(tfidf_vector_transform_test.toarray())
    # Predicted Probabilities
    y_proba = GB.predict_proba(tfidf_vector_transform_test.toarray())[:,1]


    ''' Pickle model '''
    pickle_file(GB, 'gb_model.pkl')


    ''' ROC curve and success metrics '''
    plot_roc_curve(y_test, y_proba)
    plt.show()

    tp, fp, fn, tn, cm = standard_confusion_matrix(y_test, y_pred)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    print "confusion matrix: \n{}".format(cm)
    print "recall: {}, precision: {}, f1: {}".format(recall, precision, f1)
