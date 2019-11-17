import math
import argparse

class Dataset:
  
  def __init__(self, filename):
    self.file = filename
    self.labels = []
    self.entries = []
    self.__read_file()
    self.__classify()
    self.n = len(self.labels)
    self.n_positive = len(self.positive)
    self.n_negative = len(self.negative)
    self.n_feature = len(self.entries[0])

  def __read_file(self):
    with open(self.file, 'r') as f:
      for line in f:
        data = [int(i) for i in line.split(',')]
        self.labels.append(data[0])
        self.entries.append(data[1:])
      f.close()

  def __classify(self):
    self.positive, self.negative = [], []
    for i in range(len(self.labels)):
      if self.labels[i] == 1:
        self.positive.append(self.entries[i])
      else:
        self.negative.append(self.entries[i])


class NaiveBayesian:

  def __init__(self, dataset, m=0.5):
    self.m = m
    self.training_set = dataset
    self.__learn()

  def __learn(self):
    self.feature_matrix = [[0 for _ in range(self.training_set.n_feature)], 
                           [0 for _ in range(self.training_set.n_feature)]]
    self.counts = [0, 0]
    
    for i in range(self.training_set.n):
      label = self.training_set.labels[i]
      self.counts[label] += 1
      for j in range(self.training_set.n_feature):
        if self.training_set.entries[i][j] == 1:
          self.feature_matrix[label][j] += 1

  def classify(self, testset):
    assert testset.n_feature == self.training_set.n_feature
    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0

    for i in range(testset.n_positive):
      if self.__likelihood(testset.positive[i], 1) > self.__likelihood(testset.positive[i], 0):
        true_positive += 1
      else:
        false_positive += 1
    
    for i in range(testset.n_negative):
      if self.__likelihood(testset.negative[i], 0) > self.__likelihood(testset.negative[i], 1):
        true_negative += 1
      else:
        false_negative += 1

    return true_negative, true_positive,
  
  def __likelihood(self, entry, c):
    likelihood = (math.log(self.counts[c] + self.m) 
                  - math.log(self.counts[0] + self.counts[1] + self.m))
    for i in range(len(entry)):
      s = self.feature_matrix[c][i]
      if entry[i] == 0:
        s = self.counts[c] - s
      likelihood += math.log(s + self.m) - math.log(self.counts[c] + self.m)
    return likelihood


def filename(f):
  return './data/spect-' + f +'.train.csv', './data/spect-' + f +'.test.csv' 

def print_stats(true_negative, true_positive, testset, filename):
  accuracy = (true_positive + true_negative) / testset.n
  true_negative_rate = true_negative / testset.n_negative
  true_positive_rate = true_positive / testset.n_positive
  
  print(filename, 
        str(true_positive + true_negative) + '/' + str(testset.n) + '(' + str(accuracy) + ')',
        str(true_negative) + '/' + str(testset.n_negative) + '(' + str(true_negative_rate) + ')',
        str(true_positive) + '/' + str(testset.n_positive) + '(' + str(true_positive_rate) + ')') 


choices = [
  'naive-bayesian',
]

file_choices = [
  'itg',
  'orig',
  'resplit-itg',
  'resplit'
]

parser = argparse.ArgumentParser(description="heart anomaly detector")
parser.add_argument('--learner', 
                    '-l', 
                    type=str, 
                    choices=choices, 
                    default=choices[0], 
                    help="what learner would you like to use?")
parser.add_argument('--estimator',
                    '-e',
                    type=float,
                    default=0.5,
                    help="m-estimator")
parser.add_argument('--dataset',
                    '-d',
                    type=str,
                    choices=file_choices,
                    default=file_choices[0],
                    help="choose a dataset to learn and test")

args = parser.parse_args()
fn = args.dataset
m = args.estimator

training, test = filename(fn)
f = Dataset(training)
t = Dataset(test)
n = NaiveBayesian(f, m=m)
r1, r2 = n.classify(t)
print_stats(r1, r2, t, fn)