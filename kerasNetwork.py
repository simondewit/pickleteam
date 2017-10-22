#acc: 0.3784
#download glove.twitter.27B on https://nlp.stanford.edu/projects/glove/
from __future__ import print_function

import json
import os
import numpy as np
from numpy import array

import pickle
import sys

import sklearn
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.twitter.27B/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

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

def main():
	#read documents
	documents = readFile('tweetsAsTuplesFile2.pickle')

	#create seperate lists for tweets and the categories
	texts, labels = createLists(documents)
	#test_x, test_y = createLists(test_documents)
	
	import keras
	from keras.preprocessing.text import Tokenizer
	from keras.preprocessing.sequence import pad_sequences
	from keras.utils import to_categorical
	from keras.layers import Dense, Input, Flatten
	from keras.layers import Conv1D, MaxPooling1D, Embedding
	from keras.models import Model

	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)

	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))

	data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
	labels = to_categorical(np.asarray(labels))
	print('Shape of data tensor:', data.shape)
	print('Shape of label tensor:', labels.shape)

	# split the data into a training set and a validation set
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	data = data[indices]
	labels = labels[indices]
	nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

	x_train = data[:-nb_validation_samples]
	y_train = labels[:-nb_validation_samples]
	x_val = data[-nb_validation_samples:]
	y_val = labels[-nb_validation_samples:]

	embeddings_index = {}
	f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.100d.txt'), encoding="utf8")
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

	from keras.layers import Embedding
	embedding_layer = Embedding(len(word_index) + 1,
															EMBEDDING_DIM,
															weights=[embedding_matrix],
															input_length=MAX_SEQUENCE_LENGTH,
															trainable=False)

	print('Found %s word vectors.' % len(embeddings_index))

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	x = Conv1D(128, 5, activation='relu')(embedded_sequences)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(5)(x)
	x = Conv1D(128, 5, activation='relu')(x)
	x = MaxPooling1D(35)(x)  # global max pooling
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	preds = Dense(20, activation='softmax')(x)

	model = Model(sequence_input, preds)
	model.compile(loss='categorical_crossentropy',
								optimizer='rmsprop',
								metrics=['acc'])

	model.fit(x_train, y_train, validation_data=(x_val, y_val),
						epochs=2, batch_size=128)

if __name__ == '__main__':
	main()