import numpy as np

import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from basicFunctions import BasicFunctions
from customFeatures import CustomFeatures
from customTokenizer import CustomTokenizer

from nltk.corpus import stopwords as sw

class Ensemble:
  X_train = []
  Y_train = []
  X_dev = []
  Y_dev = []
  X_test = []

  Y_predicted = []
  labels = []

  def __init__(self, X_train, X_dev, X_test, Y_train, Y_dev, labels, probabilitiesNN):
    self.X_train = X_train
    self.X_dev = X_dev
    self.X_test = X_test
    self.Y_train = Y_train
    self.Y_dev = Y_dev

    self.X = self.X_train + self.X_test
    self.Y = self.Y_train

    self.labels = labels

    self.probabilitiesNN = probabilitiesNN

  def classify(self):

    self.classifier = Pipeline([('feats', FeatureUnion([
             # ('wordCount', CustomFeatures.wordCount()),
             # ('characterCount', CustomFeatures.characterCount()),
             # ('userMentions', CustomFeatures.userMentions()),
             # ('urlMentions', CustomFeatures.urlMentions()),
             # ('instagramMentions', CustomFeatures.instagramMentions()),
             # ('hashtagUse', CustomFeatures.hashtagUse()),
	 					 # ('char', TfidfVectorizer(tokenizer=Tokenizer.tweetIdentity, lowercase=False, analyzer='char', ngram_range=(3,5), min_df=1)),#, max_features=100000)),
	 					 ('word', TfidfVectorizer(tokenizer=CustomTokenizer.tweetIdentity, lowercase=False, analyzer='word', stop_words=sw.words('english'), ngram_range=(1,20), min_df=1)),#, max_features=100000)),
      ])),
      ('classifier', SGDClassifier(loss='hinge', random_state=42, max_iter=50, tol=None))
    ])

    print("start fitting")
    self.classifier.fit(self.X_train, self.Y_train)  
    print("fitted")

  def evaluate(self):

    # print(self.labels)
    # ltest = []
    # for i in self.labels:
    #   ltest.append(self.labels.index(i))
    # print(ltest[:10])

    self.Y_predictedTEST = self.classifier.predict(self.X_test)
    print("TEST: ", self.Y_predictedTEST[:10])

    # for idx, i in np.ndenumerate(self.Y_predictedTEST):
    #   idx = idx[0]
    #   if max(self.probabilitiesNNTEST[idx]) > 0.9:  
    #     # print(i, self.Y_predictedTEST[idx], np.argmax(self.probabilitiesNNTEST[idx]))
    #     self.Y_predictedTEST[idx] = np.argmax(self.probabilitiesNNTEST[idx])

    print("TEST: ", self.Y_predictedTEST[:10])

    
    self.Y_predictedDEV = self.classifier.predict(self.X_dev)

    for idx, i in np.ndenumerate(self.Y_predictedDEV):
      idx = idx[0]
      if max(self.probabilitiesNN[idx]) > 0.95: 
        self.Y_predictedDEV[idx] = np.argmax(self.probabilitiesNN[idx])

    self.accuracy, self.precision, self.recall, self.f1score = BasicFunctions.getMetrics(self.Y_dev, self.Y_predictedDEV, self.labels)

  def printBasicEvaluation(self):    
    BasicFunctions.printEvaluation(self.accuracy, self.precision, self.recall, self.f1score, "Basic Evaluation")

  def printClassEvaluation(self):
    BasicFunctions.printClassEvaluation(self.Y_dev, self.Y_predictedDEV, self.labels)



