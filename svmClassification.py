import pickle
import sys
import numpy as np

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from sklearn import svm

#==================================================================

import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin

class CustomPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        #self.stopwords  = stopwords or set(sw.words('english'))

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        test = [list(self.tokenize(doc)) for doc in X]
        print(test)
        return test

    def tokenize(self, document):
        tokenizer = TweetTokenizer()
        tokenized_tweet = tokenizer.tokenize(document)
        # Tokenize tweet
        for token in tokenized_tweet:
            # Apply preprocessing to the token
            token = token.lower() if self.lower else token
            token = token.strip() if self.strip else token
            yield token

#==================================================================


def main():
	#read documents
	documents = readFile('tweetsAsTuplesFile2.pickle')
	
	#split documents in training and test set
	train_documents, test_documents = train_test_split(
	   documents, test_size=0.1, random_state=42)

	#create seperate lists for tweets and the categories
	train_tweets, train_categories = createLists(train_documents)
	test_tweets, test_categories = createLists(test_documents)
	
	#train the system
	#text_clf = classify(train_tweets, train_categories)
	results = classify(train_tweets, train_categories)
	
	#evaluate the system
	evaluation = evaluate(results, test_tweets, test_categories)
	print(evaluation)
	
	
def readFile(file):
	return pickle.load(open(file, 'rb'))
	
def classify(train_tweets, train_categories):
	text_clf = Pipeline([('preprocessor', CustomPreprocessor()),
						 ('vectorizer', TfidfVectorizer(tokenizer=None, preprocessor=None, lowercase=False)),
						 ('classifier', SGDClassifier(loss='hinge', penalty='l2',
											   alpha=1e-3, random_state=42,
											   max_iter=5, tol=None)),
						])
						
	text_clf.fit(train_tweets, train_categories)  
	
	return text_clf

def createLists(documents):
	tweets = []
	categories = []
	for category, tweet in documents:
		categories.append(category)
		tweets.append(tweet)
	return tweets, categories
	
def evaluate(classifier, test_tweets, test_categories):
	predicted = classifier.predict(test_tweets)
	return np.mean(predicted == test_categories)
	
if __name__ == '__main__':
	main()