from __future__ import division
import samples
import heapq
import math

class KNearestNeighbor:

    def euclideanDistance(self, x, y):
        distance = 0.0
        total = 0
        for (a, b) in zip(x, y):
            total += (a - b)**2
        distance = math.sqrt(total)
        return distance

    def classify(self, x):
        closestPoints = heapq.nsmallest(self.k, enumerate(self.trainData), key=lambda y: self.euclideanDistance(x, y[1]))
        closestLabels = [self.trainLabels[i] for (i, pt) in closestPoints]
        return max(set(closestLabels), key=closestLabels.count)

    def predict(self):
        correct = 0
        for point, label in zip(self.testData, self.testLabels):
            if self.classify(point) == label:
                correct += 1
        return correct

    def train(self, trainingData, trainingLabels, testData, testLabels, k):
        self.k = k
        self.trainData = trainingData
        self.trainLabels = trainingLabels
        self.testData = testData
        self.testLabels = testLabels        
        total = len(testLabels)

        print "Running..."
        correct = self.predict()

        ratio = 0.0
        ratio = correct / total
        print ratio
        print "Correct " + str(correct) + " Out of " + str(total)