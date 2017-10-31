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

import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.twitter.27B/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 15000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 128
epochs = 2

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

def precision(y_true, y_pred):
     """Precision metric.
 
     Only computes a batch-wise average of precision.
 
     Computes the precision, a metric for multi-label classification of
     how many selected items are relevant.
     """
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
     precision = true_positives / (predicted_positives + K.epsilon())
     return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fscore(precision,recall):
	fscore = 2 * ((precision * recall) / (precision + recall))
	return fscore


def main():
	#read documents
	documents = readFile('tweetsAsTuplesFile2.pickle')

	#create seperate lists for tweets and the categories
	texts, labels = createLists(documents)
	#test_x, test_y = createLists(test_documents)

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
								metrics=['acc',precision,recall])

	model.fit(x_train, y_train,
						epochs=epochs, batch_size=batch_size)

	score = model.evaluate(x_val, y_val, batch_size=batch_size)

	print("\n",score)

	#print('Test score:', score[0])
	#print('Test accuracy:', score[1])
	print('Test precision:', score[2])
	print('Test recall:', score[3])
	print('\nTest fscore:', fscore(score[2],score[3]))



if __name__ == '__main__':
	main()
