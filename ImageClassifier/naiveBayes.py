import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"
    self.validationLabels = validationLabels
    self.validationData = validationData
    self.kgrid = kgrid

    prob = util.Counter()

    for label in trainingLabels:
      prob[label] += 1

    self.normalize(prob)
    count = {}
    total = {}
    for feature in self.features:
      count[feature] = {0: util.Counter(), 1: util.Counter()}
      total[feature] = util.Counter()

    # ind is the index ; x are the data points with a pixel in our feature set; y is our value between 0 and 9
    for ind, x in enumerate(trainingData):
      y = trainingLabels[ind]
      
      # f is our feature point for our pixel; v is either 0 or 1 representing that pixel
      for f, v in x.items():
        count[f][v][y] += 1.0             #increment our count for label y, for feature f, value v
        total[f][y] += 1.0                #increment our total for label y for feature f

    self.smooth(count, total)


  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"
    for label in self.legalLabels:
      logJoint[label] = math.log(self.prob[label])
      for f in self.condition:
          p = self.condition[f][datum[f]][label]
          logJoint[label] += (p and math.log(p) or 0.0)

    return logJoint
  
  def normalize(self, probabilities):
    total = float(probabilities.totalCount())
    if total == 0:
        return
    for k in probabilities.keys():
      probabilities[k] = probabilities[k] / total
    self.prob = probabilities

  #Smooth our parameters
  def smooth(self, count, total):
    best_con = {}
    best_acc = None

    #Find the value of k that gives us the most accurate results
    for k in self.kgrid:
      counter = 0
      condition = {}
      for feature in self.features:
        condition[feature] = {0: util.Counter(), 1: util.Counter()}

      #Smoothing
      for feature in self.features:
        for v in [0, 1]:
          for y in self.legalLabels:
            condition[feature][v][y] = (count[feature][v][y] + k) / (total[feature][y] + k * 2)

      self.condition = condition
      guess_set = self.classify(self.validationData)

      #Run against our validation set and count how many we got correct
      for i, guess in enumerate(guess_set):
        if self.validationLabels[i] == guess:
          counter += 1.0
        else:
          counter += 0.0

      accuracy = counter / len(guess_set)         #calculate our accuracy for the given k

      #Compare accuracies and assign the best k accordingly
      if best_acc is None or accuracy > best_acc:
        best_acc = accuracy
        best_con = condition
        self.k = k

    self.condition = best_con

  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds