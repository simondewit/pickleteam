import numpy as np
import re
import os

import keras
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
    
from sklearn.model_selection import train_test_split

from basicFunctions import BasicFunctions
from customTokenizer import CustomTokenizer



class NeuralNetwork:
  word_embeddings_dir = 'glove.twitter.27B/'
  word_embeddings_dim = 200

  labels = []
  labels_dict = {}
  labels_dict_rev = {}

  avoid_skewness = False
  
  Y = []

  def __init__(self, X, Y, labels, avoid_skewness, split_amount):
    self.X = X

    self.avoid_skewness = avoid_skewness
    self.split_amount = split_amount

    self.labels = labels
    for i, label in enumerate(self.labels):
      self.labels_dict[label] = i
      self.labels_dict_rev[i] = label

    for label in Y:
      self.Y.append(self.labels_dict[label])

  def tokenize(self):
    self.X_tokenized = CustomTokenizer.tokenizeTweets(self.X) #all tweets!
    self.tokenizer = Tokenizer(split="|",)
    self.tokenizer.fit_on_texts(self.X_tokenized)
    self.sequences = self.tokenizer.texts_to_sequences(self.X_tokenized)
    self.X = pad_sequences(self.sequences)
    self.Y = to_categorical(self.Y)

  def classify(self):
    self.tokenize()

    self.X_train = self.X[:self.split_amount]
    self.Y_train = self.Y[:self.split_amount]
    self.X_test = self.X[self.split_amount:]
    self.Y_test = self.Y[self.split_amount:]

    if self.avoid_skewness:
      Y_train = np.argmax(self.Y_train, axis=1)
      Y_train = [self.labels_dict_rev[int(i)] for i in list(Y_train)]
      
      self.X_train, self.Y_train = BasicFunctions.getUnskewedSubset(self.X_train, self.Y_train, Y_train)
      self.X_train = np.array(self.X_train)
      self.Y_train = np.array(self.Y_train)

    self.createWordEmbeddings()

    self.printDataInformation()

    
    self.model = Sequential()
    self.model.add(self.word_embeddings_layer)
    self.model.add(LSTM(self.word_embeddings_dim))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(self.Y.shape[1], activation='sigmoid'))

    self.model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']) 
	
    # Train the model 
    self.model.fit(self.X_train, self.Y_train, epochs = 20, batch_size = 128, validation_split = 0.2)

  def evaluate(self):
    self.Y_predicted = self.model.predict(self.X_test)
    self.Y_predicted = np.argmax(self.Y_predicted, axis=1)
    self.Y_predicted = [self.labels_dict_rev[int(i)] for i in list(self.Y_predicted)]
    
    self.Y_test = np.argmax(self.Y_test, axis=1)
    self.Y_test = [self.labels_dict_rev[int(i)] for i in list(self.Y_test)]

    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_test, self.Y_predicted, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    BasicFunctions.printClassEvaluation(self.Y_test, self.Y_predicted, self.labels)

  def printDataInformation(self):

    print('~~~Neural Network Distribution~~~\n')
    print('Found {} unique tokens.'.format(len(self.tokenizer.word_index)))
    print('Shape of data tensor: {}'.format(self.X.shape))
    print('Shape of label tensor: {}\n'.format(self.Y.shape))

    if len(self.word_embeddings_index) > 0:
      print('Found {} word vectors.'.format(len(self.word_embeddings_index)))

  def createWordEmbeddings(self):
    self.word_embeddings_index = {}
    f = open(os.path.join(self.word_embeddings_dir, 'glove.twitter.27B.'+str(self.word_embeddings_dim)+'d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        self.word_embeddings_index[word] = coefs
    f.close()

    self.word_embeddings_matrix = np.zeros((len(self.tokenizer.word_index) + 1, self.word_embeddings_dim))
    for word, i in self.tokenizer.word_index.items():
        embedding_vector = self.word_embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            self.word_embeddings_matrix[i] = embedding_vector

    self.word_embeddings_layer = Embedding(len(self.tokenizer.word_index) + 1, self.word_embeddings_dim, mask_zero = True, weights=[self.word_embeddings_matrix], trainable = True)


    
    


