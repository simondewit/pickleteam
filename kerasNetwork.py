import pickle
import sys
import os
import numpy as np
import string

from nltk.tokenize import TweetTokenizer

import sklearn
from sklearn.metrics import classification_report, confusion_matrix

import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils import to_categorical

from createConfusionMatrix import main as mainCCM

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.twitter.27B/'
EMBEDDING_DIM = 200
batch_size = 128
epochs = 10

def main():
	#read documents
	train_documents = readFile('eng-train.pickle')
	test_documents = readFile('eng-trial.pickle')

	#create separate lists for tweets and the categories
	train_tweets, train_categories = createLists(train_documents)
	test_tweets, test_categories = createLists(test_documents)

	texts = train_tweets + test_tweets
	labels = train_categories + test_categories

	tokenizedTexts = []

	#since the Tokenizer of keras expects a text, we tokenize the tweets, but also join it together as a string
	tweetTokenizer = TweetTokenizer()
	for tweet in texts:
		tokenizedTexts.append('|'.join(tweetTokenizer.tokenize(tweet)))
	
	tokenizer = Tokenizer(split="|",)
	tokenizer.fit_on_texts(tokenizedTexts)
	sequences = tokenizer.texts_to_sequences(tokenizedTexts)

	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	data = pad_sequences(sequences)
	import ipdb; ipdb.set_trace()
	labels = to_categorical(list(map(int, labels)))
	print('Shape of data tensor:', data.shape)
	print('Shape of label tensor:', labels.shape)

	train_tweets_trans = data[0:len(train_tweets)]
	train_categories_trans = labels[0:len(train_categories)]

	test_tweets_trans = data[len(train_tweets):]
	test_categories_trans = labels[len(train_categories):]

	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.'+str(EMBEDDING_DIM)+'d.txt'), encoding="utf8")
	for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	f.close()

	embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
	for word, i in word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
					# words not found in embedding index will be all-zeros.
					embedding_matrix[i] = embedding_vector

	embedding_layer = Embedding(len(word_index) + 1, EMBEDDING_DIM, mask_zero = True, weights=[embedding_matrix], trainable = True)

	print('Found %s word vectors.' % len(embeddings_index))

	model = Sequential()
	model.add(embedding_layer)
	model.add(Dropout(0.2))
	model.add(LSTM(EMBEDDING_DIM))
	model.add(Dropout(0.2))
	model.add(Dense(20, activation='sigmoid'))

	model.compile(loss='categorical_crossentropy',
								optimizer='adam',
								metrics=['accuracy']) 

	print('Train...')
	model.fit(train_tweets_trans, train_categories_trans,
						batch_size=batch_size,
						epochs=epochs,
						validation_data=(test_tweets_trans, test_categories_trans))

	score = model.evaluate(test_tweets_trans, test_categories_trans, batch_size=batch_size)

	predicted = model.predict(test_tweets_trans)
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
	mainCCM(test_categories,predicted_categories)

	model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
	del model  # deletes the existing model

	# returns a compiled model
	# identical to the previous one
	model = load_model('my_model.h5')

def readFile(file):
	return pickle.load(open(file, 'rb'))

def createLists(documents):
	tweets = []
	categories = []
	for category, tweet in documents:
		category = int(float(category))
		categories.append([category])
		tweets.append(tweet)

	return tweets, categories

def createLists(documents):
	tweets = []
	categories = []
	for category, tweet in documents:
		categories.append(category)
		tweets.append(tweet)
	return tweets, categories


if __name__ == '__main__':
	main()
