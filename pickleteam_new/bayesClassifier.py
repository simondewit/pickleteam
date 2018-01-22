import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from basicFunctions import BasicFunctions
from customTokenizer import CustomTokenizer

class Bayes:
  X_train = []
  Y_train = []
  X_dev = []
  Y_dev = []
  X_test = []

  Y_predicted = []
  labels = []

  def __init__(self, X_train, X_dev, X_test, Y_train, Y_dev, labels):
    self.X_train = X_train
    self.X_dev = X_dev
    self.X_test = X_test
    self.Y_train = Y_train
    self.Y_dev = Y_dev

    self.X = self.X_train + self.X_test
    self.Y = self.Y_train

    self.labels = labels

  def classify(self):
    self.classifier = Pipeline([('feats', FeatureUnion([
	 					 ('char', TfidfVectorizer(tokenizer=CustomTokenizer.tweetIdentity, norm="l1", lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
	 					 ('word', TfidfVectorizer(tokenizer=CustomTokenizer.tweetIdentity, norm="l1", lowercase=False, analyzer='word', ngram_range=(1,3), min_df=1)),#, max_features=100000)),
      ])),
      ('classifier', MultinomialNB())
    ])

    print("start fitting")
    self.classifier.fit(self.X_train, self.Y_train)  
    print("fitted")

  def evaluate(self):
    self.Y_predictedTEST = self.classifier.predict(self.X_test)
    self.Y_predictedDEV = self.classifier.predict(self.X_dev)
    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_dev, self.Y_predictedDEV, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    BasicFunctions.printClassEvaluation(self.Y_dev, self.Y_predictedDEV, self.labels)

