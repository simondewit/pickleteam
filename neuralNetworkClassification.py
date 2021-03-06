import pickle
import sys
import numpy as np

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main():
	#read documents
	documents = readFile('tweetsAsTuplesFile2.pickle')
	
	#split documents in training and test set
	train_documents, test_documents = train_test_split(
	   documents, test_size=0.8, random_state=42)

	#create seperate lists for tweets and the categories
	train_tweets, train_categories = createLists(train_documents)
	test_tweets, test_categories = createLists(test_documents)
	
	#print(train_tweets[:20], train_categories[:20])
	#train the system
	text_clf = classify(train_tweets, train_categories)
	
	#evaluate the system
	evaluation = evaluate(text_clf, test_tweets, test_categories)
	print(evaluation)
	
def readFile(file):
	return pickle.load(open(file, 'rb'))
	
def classify(train_tweets, train_categories):
	text_clf = Pipeline([('vect', CountVectorizer()),
						 ('tfidf', TfidfTransformer()),
						 ('clf', MLPClassifier(solver='sgd', alpha=1e-5,	hidden_layer_sizes=(5, 3), random_state=1))])
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