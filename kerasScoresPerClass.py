import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras import metrics
from sklearn.model_selection import train_test_split
import pickle
import keras.backend as K
import numpy as np
import sklearn

from sklearn.metrics import classification_report,confusion_matrix

# from createConfusionMatrix import main as mainCCM

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #remove warnings from tensorflow

def readFile(file):
	return pickle.load(open(file, 'rb'))

def createLists(documents):
	tweets = []
	categories = []
	for category, tweet in documents:
		categories.append(category)
		tweets.append(tweet)
	return tweets, categories

def main():

	num_words 	= 10000
	epochs		= 6
	batch_size 	= 128

	#new way of splitting test and training
	train_documents = readFile('eng-train.pickle')
	test_documents = readFile('eng-trial.pickle')

	#old way of splitting test and training
	# documents = readFile('tweetsAsTuplesFileSpanish.pickle')	
	# split documents in training and test set
	# train_documents, test_documents = train_test_split(
	#    documents, test_size=0.2, random_state=42)
	# print(train_documents[:10])
	# print()
	# print(test_documents[:10])

	#create seperate lists for tweets and the categories
	train_tweets, train_categories = createLists(train_documents)
	train_categories = list(map(int, train_categories))

	test_tweets, test_categories = createLists(test_documents)
	test_categories_int = list(map(int, test_categories))

	tokenizer = Tokenizer(num_words = num_words)

	#Adapt the traing sets
	tokenizer.fit_on_texts(train_tweets)
	train_tweets = tokenizer.texts_to_sequences(train_tweets)
	train_tweets = tokenizer.sequences_to_matrix(train_tweets, mode='tfidf')
	# print('train_tweets shape:', train_tweets.shape)

	train_categories = keras.utils.to_categorical(train_categories,20)
	# print('train_categories shape:', train_categories.shape,"\n")

	#Adapt the test sets
	tokenizer.fit_on_texts(test_tweets)
	test_tweets = tokenizer.texts_to_sequences(test_tweets)
	test_tweets = tokenizer.sequences_to_matrix(test_tweets, mode='tfidf')
	# print('test_tweets shape:', test_tweets.shape)

	test_categories_positions = keras.utils.to_categorical(test_categories_int,20) #different variable needed for model.evaluate
	# print('test_categories_positions shape:', test_categories_positions.shape,"\n")


	model = Sequential()
	model.add(Dense(512, input_shape=(num_words,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(20))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])


	fit = model.fit(train_tweets, train_categories, epochs=epochs, batch_size=batch_size)
	score = model.evaluate(test_tweets, test_categories_positions, batch_size=batch_size)


	predicted = model.predict(test_tweets)
	my_labels = np.argmax(predicted, axis=1)
	predicted_categories = [str(i) for i in list(my_labels)]


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



	ourOutput = open('ourOutput.txt','wt')
	for i in predicted_categories:
		ourOutput.write(i)
		ourOutput.write("\n")


	# print(predicted_categories[:25])
	# print(test_categories[:25])

	# #creates confusion matrix
	# mainCCM(test_categories,predicted_categories)

if __name__ == '__main__':
	main()
