
import argparse
import random

import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from svmClassifier import SVM
from bayesClassifier import Bayes
from kNeighborsClassifier import KNeighbors
from decisionTreeClassifier import DecisionTree
from baselineClassifier import Baseline

from data import data

from basicFunctions import BasicFunctions

random.seed(3)

# Read arguments
parser = argparse.ArgumentParser(description='system parameters')
parser.add_argument('--method', type=str, default='svm', help='machine learning technique')
parser.add_argument('--data_method', type=int, default=1, help='how to divide the data') #only one option
parser.add_argument('--predict_languages', type=str, default='e', help='predict languages: language name or first letter of the language') #e or s
parser.add_argument('--predict_label', type=str, default='emoji', help='what to predict')
parser.add_argument('--avoid_skewness', type=bool, default=False, help='how to train the dataset, without skewness in the data or with skewness')
args = parser.parse_args()

predict_languages = BasicFunctions.getLanguages(args.predict_languages)
data = data(predict_languages)

#data.subset(50000, 5000) #to make a subset of the data (train_size, test_size)

BasicFunctions.printStandardText(args.method, predict_languages, args.predict_label)

labels = list(set(data.Y_train))
BasicFunctions.printLabelDistribution(data.Y_train)

start_time = datetime.datetime.now()

if len(labels) > 1: #otherwise, there is nothing to train
    
  if args.avoid_skewness:
    data.X_train, data.Y_train = BasicFunctions.getUnskewedSubset(data.X_train, data.Y_train)

  if args.method == 'bayes':
    classifier = Bayes(data.X_train, data.X_dev, data.X_test, data.Y_train, data.Y_dev, labels) 
  elif args.method == 'svm':
    classifier = SVM(data.X_train, data.X_dev, data.X_test, data.Y_train, data.Y_dev, labels) 
  elif args.method == 'knear':
    classifier = KNeighbors(data.X_train, data.X_dev, data.X_test, data.Y_train, data.Y_dev, labels)
  elif args.method == 'tree':
    classifier = DecisionTree(data.X_train, data.X_dev, data.X_test, data.Y_train, data.Y_dev, labels)
  elif args.method == 'neural':
    from neuralNetworkClassifier import NeuralNetwork #to avoid keras/tensorflow loading with other methods
    classifier = NeuralNetwork(data.X, data.Y, labels, args.avoid_skewness, data.split_amount)
  elif args.method == 'baseline':
    classifier = Baseline(data.X_train, data.X_dev, data.X_test, data.Y_train, data.Y_dev, labels)

  classifier.classify()
  classifier.evaluate()
  classifier.printBasicEvaluation()
  classifier.printClassEvaluation()
  BasicFunctions.writeResults(predict_languages, classifier.Y_dev, classifier.Y_predictedTEST)
  #BasicFunctions.writeConfusionMatrix(classifier.Y_test, classifier.Y_predicted)

  end_time = datetime.datetime.now()
  BasicFunctions.printDuration(start_time, end_time)

  
else:
  print('The combination of the language <{}> and the variable <{}> only have one label. Thus, there is nothing to train. Try another combination!'.format(predict_languages, args.predict_label))