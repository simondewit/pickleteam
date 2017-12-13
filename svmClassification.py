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
from nltk.stem import WordNetLemmatizer 
import nltk

#from createConfusionMatrix import main as mainCCM
from nltk.stem import SnowballStemmer, PorterStemmer, LancasterStemmer
from nltk.stem.lancaster import LancasterStemmer


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

	ourOutput = open('ourOutputEnglish.txt','wt')
	for i in predicted_categories:
		ourOutput.write(i)
		ourOutput.write("\n")

	goldOutput = open('goldOutputEnglish.txt','wt')
	for i in test_categories:
		goldOutput.write(i)
		goldOutput.write("\n")

	# ourOutput = open('ourOutputSpanish.txt','wt')
	# for i in predicted_categories:
	# 	ourOutput.write(i)
	# 	ourOutput.write("\n")

	# goldOutput = open('goldOutputSpanish.txt','wt')
	# for i in test_categories:
	# 	goldOutput.write(i)
	# 	goldOutput.write("\n")

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg

def customStemmer(arg):
	"""
    Preprocesser function to test different stemmers.
    """
	#st = SnowballStemmer('english')
	st = PorterStemmer()
	#st = LancasterStemmer()
	return st.stem(arg) 

def customLemmatizer(arg):
	"""
    Preprocesser function to test different lemma.
    """
	wnl = WordNetLemmatizer()
	st = PorterStemmer()
	return st.stem(wnl.lemmatize(arg))

def tweetIdentity(arg):
	# tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	# return tokenizer.tokenize(arg)

	tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
	tokens = tokenizer.tokenize(arg)
	return [token+"_POS-"+tag for token, tag in nltk.pos_tag(tokens)]
	
def readFile(file):
	return pickle.load(open(file, 'rb'))
	
def classify(train_tweets, train_categories):
	#('preprocessor', CustomPreprocessor()),
	text_clf = Pipeline([#[('feats', FeatureUnion([
						 #('char', TfidfVectorizer(tokenizer=tweetIdentity, norm="l1", preprocessor=CustomPreprocessor, lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
						 ('word', TfidfVectorizer(tokenizer=tweetIdentity, norm="l1", preprocessor=customLemmatizer, stop_words=sw.words('english'), lowercase=False, analyzer='word', ngram_range=(1,3), min_df=1)),#, max_features=100000)),
						 #])),
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
