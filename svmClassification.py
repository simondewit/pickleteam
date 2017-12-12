import pickle
import sys
import numpy as np
import string

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from nltk.tokenize import TweetTokenizer
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

#from createConfusionMatrix import main as mainCCM
from nltk.stem import SnowballStemmer


def main():
	#read documents
	train_documents = readFile('eng-train.pickle')
	test_documents = readFile('eng-trial.pickle')

	#create seperate lists for tweets and the categories
	train_tweets, train_categories = createLists(train_documents)
	test_tweets, test_categories = createLists(test_documents)
	
	#train the system
	#text_clf = classify(train_tweets, train_categories)
	results = classify(train_tweets, train_categories)
	
	#evaluate the system
	evaluation, predicted_categories = evaluate(results, test_tweets, test_categories)
	print(evaluation)


	#average scores
	precisionScoreAverage = sklearn.metrics.precision_score(test_categories,predicted_categories, average="macro")
	print("\n\nprecision sklearn:", round(precisionScoreAverage,3))

	recallScoreAverage = sklearn.metrics.recall_score(test_categories,predicted_categories, average="macro")
	print("recall sklearn:", round(recallScoreAverage,3))

	f1ScoreAverage = sklearn.metrics.f1_score(test_categories,predicted_categories, average="macro")
	print("fscore sklearn:", round(f1ScoreAverage,3),"\n")


	#scores per class
	labels = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]
	print("Class \t Precision \t Recall \t F-score")
	for label in labels:
		precisionScore = sklearn.metrics.precision_score(test_categories,predicted_categories, average="macro", labels=label)
		recallScore = sklearn.metrics.recall_score(test_categories,predicted_categories, average="macro", labels=label)
		f1Score = sklearn.metrics.f1_score(test_categories,predicted_categories, average="macro", labels=label)

		print(label, "\t", round(precisionScore,3), "\t\t", round(recallScore,3), "\t\t", round(f1Score,3), "\t")


	print()
	#create confusion matrix
	# mainCCM(test_categories,predicted_categories) #both variables must be lists
	
# class CustomPreprocessor(BaseEstimator, TransformerMixin):

#     def __init__(self, stopwords=None, punct=None,
#                  lower=True, strip=True):
#         self.lower      = lower
#         self.strip      = strip
#         # self.stopwords  = stopwords or set(sw.words('spanish'))

#     def fit(self, X, y=None):
#         return self

#     def inverse_transform(self, X):
#         return [" ".join(doc) for doc in X]

#     def transform(self, X):
#         return [list(self.tokenize(doc)) for doc in X]

#     def tokenize(self, document):
#         tokenizer = TweetTokenizer()
#         tokenized_tweet = tokenizer.tokenize(document)
#         # Tokenize tweet
#         for token in tokenized_tweet:
#             # Apply preprocessing to the token
#             token = token.lower() if self.lower else token
#             token = token.strip() if self.strip else token
#             yield token
	
def CustomPreprocessor(arg):
	# print(arg)
	argListNew = []
	argList = arg.split()
	for elem in argList:
		if elem not in sw.words('spanish'):
			argListNew.append(elem)
	arg = " ".join(argListNew)
	return arg


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

def customStemmer(arg):
	"""
    Preprocesser function to test different stemmers.
    """
	stm = SnowballStemmer('english')
	return stm.stem(arg) 

def tweetIdentity(arg):
	tokenizer = TweetTokenizer()
	return tokenizer.tokenize(arg)
	
def readFile(file):
	return pickle.load(open(file, 'rb'))
	
def classify(train_tweets, train_categories):
	#('preprocessor', CustomPreprocessor()),
	text_clf = Pipeline([('feats', FeatureUnion([
						 ('char', TfidfVectorizer(tokenizer=tweetIdentity, norm="l1", preprocessor=None, preprocessor=CustomPreprocessor, lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
						 ('word', TfidfVectorizer(tokenizer=tweetIdentity, norm="l1", preprocessor=None, preprocessor=CustomPreprocessor, lowercase=False, analyzer='word', ngram_range=(1,3), min_df=1)),#, max_features=100000)),
						 ])),
						 ('classifier', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=None))])
	

	text_clf.fit(train_tweets, train_categories)  
	return text_clf

def createLists(documents):
	tweets = []
	categories = []
	for category, tweet in documents:
		categories.append(category)
		if "…" in tweet:
			tweet = tweet.replace("…","...")
		tweets.append(tweet)

	return tweets, categories
	
def evaluate(classifier, test_tweets, test_categories):
	predicted = classifier.predict(test_tweets)
	return np.mean(predicted == test_categories), predicted
	
if __name__ == '__main__':
	main()
