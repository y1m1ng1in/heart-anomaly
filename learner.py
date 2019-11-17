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
    self.dataset = dataset
    self.__feature_prob()
    self.__feature_prob_given_negative()
    self.__feature_prob_given_positive()
    self.__positive_prob()
    self.__negative_prob()
    print(self.p_entry, self.n_entry)

  def __feature_prob(self):
    self.p = self.__probabilities(self.dataset.entries)
    self.n = [1-p for p in self.p]

  def __feature_prob_given_positive(self):
    self.p_given_p = self.__probabilities(self.dataset.positive)
    self.n_given_p = [1-p for p in self.p_given_p]

  def __feature_prob_given_negative(self):
    self.p_given_n = self.__probabilities(self.dataset.negative)
    self.n_given_n = [1-p for p in self.p_given_n]

  def __positive_prob(self):
    self.p_entry = self.dataset.n_positive / self.dataset.n

  def __negative_prob(self):
    self.n_entry = self.dataset.n_negative / self.dataset.n

  def __probabilities(self, entries):
    assert entries
    n_entries = len(entries)
    result = [0 for _ in range(self.dataset.n_feature)]
    for i in range(n_entries):
      for j in range(self.dataset.n_feature):
        result[j] += entries[i][j]
    for i in range(self.dataset.n_feature):
      result[i] /= n_entries
      assert result[i] >= 0 and result[i] <= 1
    return result
    

f = Dataset('./data/spect-itg.train.csv')
n = NaiveBayesian(f)