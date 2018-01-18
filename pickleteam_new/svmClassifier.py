import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from basicFunctions import BasicFunctions
from customFeatures import CustomFeatures
from customTokenizer import CustomTokenizer

from nltk.corpus import stopwords as sw

class SVM:
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []

  Y_predicted = []
  labels = []

  def __init__(self, X_train, X_test, Y_train, Y_test, labels):
    self.X_train = X_train
    self.X_test = X_test
    self.Y_train = Y_train
    self.Y_test = Y_test

    self.X = self.X_train + self.X_test
    self.Y = self.Y_train + self.Y_test

    self.labels = labels

  def classify(self):

    self.classifier = Pipeline([('feats', FeatureUnion([
             #('wordCount', CustomFeatures.wordCount()),
             #('characterCount', CustomFeatures.characterCount()),
            #  ('userMentions', CustomFeatures.userMentions()),
            #  ('urlMentions', CustomFeatures.urlMentions()),
            #  ('instagramMentions', CustomFeatures.instagramMentions()),
             #('hashtagUse', CustomFeatures.hashtagUse()),
	 					 #('char', TfidfVectorizer(tokenizer=Tokenizer.tweetIdentity, lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
	 					 ('word', TfidfVectorizer(tokenizer=CustomTokenizer.tweetIdentity, lowercase=False, analyzer='word', stop_words=sw.words('english'), ngram_range=(1,20), min_df=1)),#, max_features=100000)),
      ])),
      ('classifier', SGDClassifier(loss='hinge', random_state=42, max_iter=50, tol=None))
    ])

    self.classifier.fit(self.X_train, self.Y_train)  

  def evaluate(self):
    self.Y_predicted = self.classifier.predict(self.X_test)
    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_test, self.Y_predicted, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    BasicFunctions.printClassEvaluation(self.Y_test, self.Y_predicted, self.labels)


