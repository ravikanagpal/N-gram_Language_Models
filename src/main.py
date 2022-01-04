# This program should loop over the files listed in the input file directory,
# assign perplexity with each language model,
# and produce the output as described in the assignment description.


# import dependencies
import os
import sys
import re
import collections
from collections import Counter
import nltk
from nltk.util import ngrams
import math
import numpy as np
import pandas as pd


# getting the training data
def get_train_data_files(directory):
    textfiles = []
    for filename in os.listdir(directory):
        # getting the files ending with .txt.tra (training files)
        if filename.endswith(".txt.tra"):
            textfiles.append(filename)
        else:
            continue
    return textfiles


# obtaining development data from the data/dev folder. 
def get_dev_data_files(directory):
    textfiles_dev = []
    for filename in os.listdir(directory):
        # getting the files ending with .txt.dev (development files)
        if filename.endswith(".txt.dev"):
            textfiles_dev.append(filename)
        else:
            continue
    return textfiles_dev


# getting the data from the files
def get_data(filename, directory):
    file_name = directory + filename
    script = ""
    with open(file_name, 'r+') as f:
        text = f.readlines()
        # in each line of the text, appending the start and end token (each sentence)
        for line in text:
            line = "<s> " + line + " </s> "
            script = script + line
        # replacing the "\n" token  after each sentences since we have appended start and end token
        text = script.replace("\n", "")

        return text


# saving the results in the csv format
def save_results(results, resultfile):
    column_names = ["Training_file", "Testing_file", "Perplexity", "N"]
    df = pd.DataFrame(results, columns=column_names)
    # storing the results to csv file
    df.to_csv(resultfile, index=False)
    return df


# preprocessing tokens
def preprocessing_tokens(text):
    """
    This method preprocess the tokens (add the <UNK> token based on the frequency)

    Parameters
    text : String
        Text is passed as an argument that needs to be preprocessed.

    Returns
    new_tokens:list
        list of new_tokens after the preprocessing
    """
    tokens = []
    for line in text:
        word = line.split()
        for w in word:
            for c in w:
                tokens.append(c)

    token_counts = Counter(tokens)

    new_tokens = []
    for t in tokens:
        # checking if the frequency of the token count is 1, then it will append <UNK> token
        if token_counts[t] == 1:
            new_tokens.append("<UNK>")
        else:
            new_tokens.append(t)

    return new_tokens


"""# Unsmoothed mechanism"""


class UnsmoothedModel(object):
    """
    This class implements the Unsmoothed Model and inherits the Object Python class

    Methods
    __init__
        Constructor that sets important variables of class like vocab, freq, context, tokens, text.
    unsmoothed
        this method will train the model by building the n-gram model, finding the frequency of n-grams and the counts of the context

    """

    def __init__(self, n, data):
        self.n = n
        self.text = data
        self.vocabulary = []
        self.count = None
        self.context = None
        self.tokens = preprocessing_tokens(self.text)

    # utility method to find the context
    def find_ccount(self, N):
        """
        This method finds the context of the model and the return the count based on its context

        Parameters
        N: int
          It holds the length of the tokens

        Returns
        None
        """
        if self.n == 1:
            # if n== 1 , the context will be the length of  the Vocab
            self.context = N
        else:
            c = ngrams(self.tokens, self.n - 1)
            self.context = Counter(c)

    # implementation for unsmoothed model
    def unsmoothed(self):
        """
        This trains the unsmoothed model by building the n-gram language model, finding its vocabulary,

        Parameters
        None

        Returns
        n:int
            the value for n-gram model
        vocabulary: list
            the list holds the unique characters
        count: Counter Object (dict)
            returns the count of n-gram model values
        context: Counter object (dict) if n>1 | int if n==1
            return the count on the context values. It will return the length of tokens in case n==1 else return the counter for n-1 gram 
        """

        if '<UNK>' not in self.tokens:
            self.tokens.append("<UNK>")
        N = len(self.tokens)
        self.vocabulary = list(set(self.tokens))
        # finding ngram model using nltk
        model = ngrams(self.tokens, self.n)
        self.count = Counter(model)
        self.find_ccount(N)

        return self.n, self.vocabulary, self.count, self.context

    # finding perplexity for test data
    # data source: https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954
    def find_perplexity(self, text, n, Vocab, count, context):
        """
        This method takes the test data, first generate the n-gram model of text, then find the preplexity of test data on our trained model.

        Parameters
        text: String
          The test text, in which we we want to test our model
        n: int
          n refers to the model we want to create, example: unigram, bigram....
        Vocab: list
           list containing unique vocabulary (list of characters)
        count: Dict
           it contains the count of each value (n-gram) model .
        context: Counter object (dict) if n>1 | int if n==1
          the count on the context values.

        Returns
        perplexity: float
          returns perplexity of our model
        """
        # obtaining the tokens from the text
        t_tokens = []
        for line in text:
            word = line.split()
            for w in word:
                for c in w:
                    t_tokens.append(c)
        # appending the "<UNK> token in case my vocab does not have this token"
        new_t_tokens = []
        for tt in t_tokens:
            if tt in Vocab:
                new_t_tokens.append(tt)
            else:
                new_t_tokens.append("<UNK>")
        # finding the probability of the model
        probability, log_probability = 0, 0
        # to find the denominator
        model = ngrams(new_t_tokens, n)
        for item in model:
            # unigram model (the context is N: length of token)
            if n == 1:
                log_probability = np.log2(count[item] / context)
            # for n-gram model where n>=2
            else:
                if (context[item[:-1]] == 0):
                    log_probability = 0
                else:
                    log_probability = np.log2(count[item] / context[item[:-1]])

            probability = probability + log_probability
        perplexity = np.power(2, (-1 / len(new_t_tokens)) * probability)
        return perplexity


"""# Laplace mechanism"""


class LaplaceModel(object):

  def __init__(self, n, data):
    self.n = n
    self.text = data
    self.vocabulary = []
    self.count = None
    self.context = None
    self.tokens = preprocessing_tokens(self.text)
    self.len_tok = None

  def find_ccount(self):
    # finding the context
    """
    This method finds the context of the model and the return the count based on its context
    
    Parameters
    None

    Returns
    None
    """
    c = ngrams(self.tokens, self.n-1)
    self.context = Counter(c)

  def laplace(self):
    """
    This trains the unsmoothed model by building the n-gram language model, finding its vocabulary, 
    
    Parameters
    None

    Returns
    n:int    
        the value for n-gram model
    vocabulary: list
        the list holds the unique characters
    count: Counter Object (dict)
        returns the count of n-gram model values 
    context: Counter object (dict) if n>1 | int if n==1
        return the count on the context values. 
    """
    # adding token "<UNK>" if not in my token list
    if '<UNK>' not in self.tokens:
      self.tokens.append("<UNK>")
    self.len_tok = len(self.tokens)
    self.vocabulary = list(set(self.tokens))
    # generating an ngram model
    model = ngrams(self.tokens, self.n)
    self.count = Counter(model)
   
    self.find_ccount()
    return self.n, self.vocabulary, self.count, self.context

  # finding perplexity for test data
  # data source: https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954
  def find_perplexity(self, text, n, Vocab, count, context):
    """
    This method takes the test data, first generate the n-gram model of text, then find the preplexity of test data on our trained model.

    Parameters
    text: String
        The test text, in which we we want to test our model
    n: int
        n refers to the model we want to create, example: unigram, bigram....
    Vocab: list
        list containing unique vocabulary (list of characters)
    count: Dict
        it contains the count of each value (n-gram) model .
    context: Counter object (dict) if n>1 | int if n==1
        the count on the context values.

    Returns
    perplexity: float
        returns perplexity of our model
    """
    # getting each token from the text
    t_tokens = []
    for line in text:
      word = line.split()
      for w in word:
        for c in w:
          t_tokens.append(c)
    new_t_tokens = []
    # adding "<UNK> " token if the token is not in the Vocab
    for tt in t_tokens:
      if tt in Vocab:
        new_t_tokens.append(tt)
      else:
        new_t_tokens.append("<UNK>")
    # intializing default probability to 0
    probability, log_probability = 0, 0
    # to find the denominator
    model = ngrams(new_t_tokens, n)
    for item in model:
      # adding laplace smoothing : adding  1 to numerator
      # adding laplace smoothing to denominator by adding the len(Vocab) of n-gram
      if n == 1 :
        # for unigram the context is N: length of tokens
        log_probability = np.log2((count[item]+1)/(self.len_tok + len(Vocab)))
      else:
        # based on this data source: https://www.cs.utexas.edu/~mooney/cs388/slides/equation-sheet.pdf
        log_probability = np.log2((count[item]+1)/(context[item[:-1]] + len(context)))
      # calculating log probability
      #print(item, context, numer, denom, log_probability)
      probability = probability + log_probability
    # using the preplexity formula 
    perplexity = np.power(2, (-1/len(new_t_tokens))* probability)
    return perplexity



"""## **Interpolation Model**"""


class InterpolationModel(object):

    def __init__(self, n, data):
        self.n = n
        self.text = data
        self.vocabulary = []
        self.count = None
        self.m_grams = dict()
        self.lambdas = None
        self.tokens = preprocessing_tokens(self.text)
        self.N = len(self.tokens)
        self.ratios = self.initialize_zeros(n)
        self.probabilities = self.initialize_zeros(n)

    def initialize_zeros(self, n):
        l = []
        for i in range(n + 1):
            l.append(0.0)
        return l

    def deleted_interpolation(self):
        """
        This trains the unsmoothed model by building the n-gram language model, finding its vocabulary,

        Parameters
        None

        Returns
        n:int
            the value for n-gram model
        vocabulary: list
            the list holds the unique characters
        lambdas: list
           list of weights required for each n-grams
        m_grams: dictionary containing list
            dictionary containing list of counters
        """

        if '<UNK>' not in self.tokens:
            self.tokens.append("<UNK>")

        self.vocabulary = list(set(self.tokens))
        model = ngrams(self.tokens, self.n)
        self.count = Counter(model)
        # capture ngrams for each of the values till n in m_grams.

        for t in range(self.n):
            self.m_grams[t] = Counter(ngrams(self.tokens, t+1))
        # calculate lambdas to be multiplied with the proabilities of different n_grams
        self.lambdas = self.calculate_lambdas(self.m_grams, self.n,  self.N)
        return self.n, self.lambdas, self.m_grams, self.vocabulary

    def calculate_lambdas(self, m_grams, n, N):
        """
        This functions calculate the lambas for interpolation part

        Parameters
        n: int
            n-gram model
        m_grams: dictionary containing list
            dictionary containing list of counters
        N: int
            length of tokens

        Returns
        self.normalize_lambas: function
            this function returns the list of normalized lambdas
        """

        lambdas = []
        for i in range(n):
            lambdas.append(0.0)
        for value in m_grams[n - 1]:
            if (m_grams[n - 1][value] > 0):
                c_value = value
                hold_ratios = []
                for i in range(n):
                    hold_ratios.append(0.0)
                for j in range(n):
                    # print(j, c_value)
                    if (j != n - 1):
                        num = m_grams[len(c_value) - 1][c_value] - 1
                        den = m_grams[len(c_value) - 2][c_value[:-1]] - 1
                    else:
                        num = m_grams[len(c_value) - 1][c_value] - 1
                        den = N - 1
                    if (den != 0):
                        hold_ratios[j] = num / den
                    # print(hold_ratios)
                    c_value = c_value[:-1]
                max_value = max(hold_ratios)
                max_index = hold_ratios.index(max_value)
                lambdas[n - max_index - 1] += m_grams[n - 1][value]
        return self.normalize_lambdas(lambdas)

    def normalize_lambdas(self, lambdas):
        """
        This functions calculate the lambas for interpolation part

        Parameters
        lambdas: list
           list of weights required for each n-grams

        Returns
        arr: list
            this return a list of normalized lambdas
        """

        sumratio = 0.0
        for item in lambdas:
            sumratio = sumratio + item
        arr = []
        for value in lambdas:
            arr.append(float(value) / float(sumratio))
        return arr

    def findInterpolatedProbability(self, tokens, n, m_grams, lambdas):
        """
        This functions finds the probability of interpolation model

        Parameters
        tokens: list
            tokens in the test file
        n: int
            n-gram model
        m_grams: dictionary containing list
            dictionary containing list of counters
        lambdas: list
           list of weights required for each n-grams

        Returns
        log_probability: float
            return the log probability
        
        """
        log_probability = 0
        # to find the denominator
        model = ngrams(tokens, n)
        for item in model:
            probability = 0
            current_item = item
            for t in range(n):
                cprob = 0.0
                if len(current_item) == 1:
                    cprob = m_grams[len(current_item)-1][current_item] / self.N
                else:
                    if (m_grams[len(current_item) - 2][current_item[:-1]] > 0):
                        cprob = m_grams[len(current_item)-1][current_item] / m_grams[len(current_item) - 2][current_item[:-1]]
                    else:
                        cprob = 0.0
                probability = probability + lambdas[len(current_item) - 1] * cprob
                current_item = current_item[1:]

            log_probability = log_probability + np.log2(probability)
        return log_probability

    # finding perplexity for test data
    # data source: https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk/55043954
    def find_perplexity(self, test_data, n, lambdas, m_grams, Vocab):
        """
        This method takes the test data, first generate the n-gram model of text, then find the preplexity of test data on our trained model.

        Parameters
        test_data: String
          The test text, in which we we want to test our model
        n: int
          n refers to the model we want to create, example: unigram, bigram....
        lambdas: list
           list of weights required for each n-grams    
        m_grams: dictionary containing list
            dictionary containing list of counters
        Vocab: list
           list containing unique vocabulary (list of characters)

        Returns
        perplexity: float
          returns perplexity of our model
        """
        t_tokens = []
        for line in test_data:
            word = line.split()
            for w in word:
                for c in w:
                    t_tokens.append(c)
        new_t_tokens = []
        for tt in t_tokens:
            if tt in Vocab:
                new_t_tokens.append(tt)
            else:
                new_t_tokens.append("<UNK>")
        probability = self.findInterpolatedProbability(new_t_tokens, n, m_grams, lambdas)
        perplexity = np.power(2, (-1 / len(new_t_tokens)) * probability)
        return perplexity


#### utility methods ################
def train_unsmoothed(textfiles_train, train_dir, unsmoothed_n=1):
    # this method will train the unsmoothed model
    models_values = []
    for file_name in textfiles_train:
        text = get_data(file_name, train_dir)
        model = UnsmoothedModel(unsmoothed_n, text)
        n, Vocab, freq, context = model.unsmoothed()
        models_values.append((model, file_name, n, Vocab, freq, context))
    return models_values


def train_laplace(textfiles_train, train_dir, laplace_n=3):
    # this method will train the laplace model
    models_laplace = []
    for file_name in textfiles_train:
        text = get_data(file_name, train_dir)
        model = LaplaceModel(laplace_n, text)
        n, Vocab, freq, context = model.laplace()
        models_laplace.append((model, file_name, n, Vocab, freq, context))
    return models_laplace


def train_interpolation(textfiles_train, train_dir, interp_n=6):
    # this method will train the deleted interpolation model
    models_values = []
    for file_name in textfiles_train:
        fname = os.path.join(train_dir, file_name)
        with open(fname, 'r+') as f:
            text = get_data(file_name, train_dir)
            model = InterpolationModel(interp_n, text)
            n, lambdas, m_grams, Vocab = model.deleted_interpolation()
            models_values.append((model, file_name, n, lambdas, m_grams, Vocab))
    return models_values


# testing on the development set and finding the training file that generates the best guess
# getting the development data
def testmodel(models_values, textfiles_dev, test_dir, output_file):
    results1 = []
    for index, file_name in enumerate(textfiles_dev):
        text = get_data(file_name, test_dir)
        perp = []
        for index, model in enumerate(models_values):
            # finding the perplexity on each model
            perp.append(model[0].find_perplexity(text, model[2], model[3], model[4], model[5]))
        # finding the minimum perplexity to find the best model
        min_perplex = min(perp)
        # getting the best model by checking the index which generate minimum perplexity
        best_model = models_values[perp.index(min_perplex)][1]

        results1.append((best_model, file_name, min_perplex, N))
        # to sort based on dev set file
        # data source: https://stackoverflow.com/questions/3121979/how-to-sort-a-list-tuple-of-lists-tuples-by-the-element-at-a-given-index
        results1.sort(key=lambda tup: tup[1])  # sorts in place
        save_results(results1, output_file)


## main method##
train_dir = sys.argv[1]
textfiles_train = get_train_data_files(train_dir)

test_dir = sys.argv[2]
textfiles_dev = get_dev_data_files(test_dir)

output_file = sys.argv[3]
if (len(sys.argv) > 4):
    optional_arg = sys.argv[4]
else:
    print("No optional argument passed. Pass an optional argument to choose a model to run. ")
    exit()

# calling different methods with optional arguments
models_values = []
N = None
if optional_arg == "--unsmoothed":
    N = 1
    models_values = train_unsmoothed(textfiles_train, train_dir, N)
    testmodel(models_values, textfiles_dev, test_dir, output_file)
elif optional_arg == "--laplace":
    N = 3
    models_values = train_laplace(textfiles_train, train_dir, N)
    testmodel(models_values, textfiles_dev, test_dir, output_file)
elif optional_arg == "--interpolation":
    N = 3
    models_values = train_interpolation(textfiles_train, train_dir, N)
    testmodel(models_values, textfiles_dev, test_dir, output_file)
