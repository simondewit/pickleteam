import os
import xml.etree.ElementTree as ET

import copy

import re
import pickle 

import random

class data:

  X_train = []
  Y_train = []
  X_dev = []
  Y_dev = []
  X_test = []

  def __init__(self, language):
    if language == ['english']:
      self.files = {'train': 'eng-train.pickle', 
                    'development': 'eng-trial.pickle',
                    'test': 'us_test.pickle' }
    elif language == ['spanish']:
      self.files = {'train': 'es-train.pickle', 
                    'development': 'es-trial.pickle',
                    'test': 'es_test.pickle' }

    self.readFiles()

  def readFile(self, file):
    return pickle.load(open(self.files[file], 'rb'))

  def transform(self):
    for Y, X in self.data_train:

      X = re.sub(r'\…', ' $INSTAGRAM$', X)
      X = re.sub(r'http:\/\/t.co\S*', ' %URL% ', X)
      X = re.sub(r'https:\/\/t.co\S*', ' %URL% ', X)

      X = re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)
      X = re.sub(r'(.)\1{3,}', r'\1\1\1', X)
      
      self.X_train.append(X)
      self.Y_train.append(Y)

    for Y, X in self.data_dev:

      X = re.sub(r'\…', ' $INSTAGRAM$', X)
      X = re.sub(r'http:\/\/t.co\S*', ' %URL% ', X)
      X = re.sub(r'https:\/\/t.co\S*', ' %URL% ', X)
      
      X = re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)
      X = re.sub(r'(.)\1{3,}', r'\1\1\1', X)

      self.X_dev.append(X)
      self.Y_dev.append(Y)

    for X in self.data_test:

      X = re.sub(r'\…', ' $INSTAGRAM$', X)
      X = re.sub(r'http:\/\/t.co\S*', ' %URL% ', X)
      X = re.sub(r'https:\/\/t.co\S*', ' %URL% ', X)
      
      X = re.sub(r'[^a-zA-Z0-9 #$%&]', ' ', X)
      X = re.sub(r'(.)\1{3,}', r'\1\1\1', X)

      self.X_test.append(X)

    self.X = copy.copy(self.X_train)
    self.Y = copy.copy(self.Y_train)

    self.X.extend(self.X_test)
    self.Y.extend(self.Y_test)

    self.split_amount = len(self.X_train)

  def readFiles(self):
    self.data_train = self.readFile('train')
    self.data_dev = self.readFile('development')
    self.data_test = self.readFile('test')

    self.transform()

  def subset(self, train_amount, test_amount):
    self.X_train, self.Y_train = self.subsetBase(self.X_train, self.Y_train, train_amount)
    self.X_test, self.Y_test = self.subsetBase(self.X_test, self.Y_test, test_amount)

  def subsetBase(self, listA, listB, amount):
    all_keys = []
    for i, val in enumerate(listA):
      all_keys.append(i)

    new_listA = []
    new_listB = []

    subset_keys = sorted(random.sample(all_keys, amount))

    for key in subset_keys:
      new_listA.append(listA[key])
      new_listB.append(listB[key])

    return new_listA, new_listB



  

