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

from createConfusionMatrix import main as mainCCM


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

	num_words = 5000
	epochs = 1
	batch_size = 128

	documents = readFile('tweetsAsTuplesFile2.pickle')
		
	# split documents in training and test set
	train_documents, test_documents = train_test_split(
	   documents, test_size=0.2, random_state=42)

	#create seperate lists for tweets and the categories
	train_tweets, train_categories = createLists(train_documents)
	train_categories = list(map(int, train_categories))

	test_tweets, test_categories = createLists(test_documents)
	# print("\n1\n",test_categories[:10],"\n")
	test_categories_int = list(map(int, test_categories))
	# print("\n2\n",test_categories[:10],"\n")

	tokenizer = Tokenizer(num_words = num_words)

	#Adapt the traing sets
	tokenizer.fit_on_texts(train_tweets)
	train_tweets = tokenizer.texts_to_sequences(train_tweets)
	train_tweets = tokenizer.sequences_to_matrix(train_tweets, mode='tfidf')
	print('train_tweets shape:', train_tweets.shape)

	train_categories = keras.utils.to_categorical(train_categories,20)
	print('train_categories shape:', train_categories.shape,"\n")

	#Adapt the test sets
	tokenizer.fit_on_texts(test_tweets)
	test_tweets = tokenizer.texts_to_sequences(test_tweets)
	test_tweets = tokenizer.sequences_to_matrix(test_tweets, mode='tfidf')
	print('test_tweets shape:', test_tweets.shape)

	test_categories_positions = keras.utils.to_categorical(test_categories_int,20)
	print('test_categories_positions shape:', test_categories_positions.shape,"\n")


	model = Sequential()
	model.add(Dense(512, input_shape=(num_words,)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(20))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy',precision,recall])


	fit = model.fit(train_tweets, train_categories, epochs=epochs, batch_size=batch_size)
	score = model.evaluate(test_tweets, test_categories_positions, batch_size=batch_size)


	predicted = model.predict(test_tweets) 
	my_labels = np.argmax(predicted, axis=1)
	predicted_categories = [str(i) for i in list(my_labels)]

	# #print('Test score:', score[0])
	# #print('Test accuracy:', score[1])
	print('\nTest precision keras:', score[2])
	print('Test recall keras:', score[3])
	print('Test fscore keras:', fscore(score[2],score[3]))

	metricsPerClass = classification_report(test_categories, predicted_categories)
	print(metricsPerClass)

	
	# #creates confusion matrix
	# mainCCM(test_categories,predicted_categories) #hier moet predicted dus alleen no ingevuld worden met een lijst van de gevonden labels

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


if __name__ == '__main__':
	main()
