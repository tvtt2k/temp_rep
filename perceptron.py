
import numpy as np

from binary import BinaryClassifier
import util

class Perceptron(BinaryClassifier):

    def __init__(self, opts):
        
        BinaryClassifier.__init__(self, opts)
        self.opts = opts
        self.reset()

    def reset(self):

        self.weights = 0    # our weight vector
        self.bias    = 0    # our bias
        self.numUpd  = 0    # number of updates made

    def online(self):
        """
        Our perceptron is online
        """
        return True

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return    "w=" + repr(self.weights)   +  ", b=" + repr(self.bias)

    def predict(self, X):
        """
        X is a vector that we're supposed to make a prediction about.
        Our return value should be the margin at this point.
        Semantically, a return value <0 means class -1 and a return
        value >=0 means class +1
        """

        if self.numUpd == 0:
            return 0          # failure
        else:
            return np.dot(self.weights, X) + self.bias   # this is done for you!

    def nextExample(self, X, Y):
        """
        X is a vector training example and Y is its associated class.
        We're guaranteed that Y is either +1 or -1.  We should update
        our weight vector and bias according to the perceptron rule.
        """

        # check to see if we've made an error
        if Y * self.predict(X) <= 0:   ### SOLUTION-AFTER-IF
            self.numUpd  = self.numUpd  + 1

            # perform an update
            self.weights = self.weights + Y*X### TODO: YOUR CODE HERE

            self.bias    = self.bias + Y    ### TODO: YOUR CODE HERE

    def nextIteration(self):
        """
        Indicates to us that we've made a complete pass through the
        training data.  This function doesn't need to do anything for
        the perceptron, but might be necessary for other classifiers.
        """
        return   # don't need to do anything here

    def getRepresentation(self):
        """
        Return a tuple of the form (number-of-updates, weights, bias)
        """

        return (self.numUpd, self.weights, self.bias)

    def train(self, X, Y):
        """
        (BATCH ONLY)

        X is a matrix of data points, Y is a vector of +1/-1 classes.
        """
        if self.online():
            for epoch in range(self.opts['numEpoch']):
                # loop over every data point
                for n in range(X.shape[0]):
                    # supply the example to the online learner
                    self.nextExample(X[n], Y[n])

                # tell the online learner that we're
                # done with this iteration
                self.nextIteration()
        else:
            util.raiseNotDefined()


class PermutedPerceptron(Perceptron):

    def train(self, X, Y):
        """
        (BATCH ONLY)

        X is a matrix of data points, Y is a vector of +1/-1 classes.
        """
        if self.online():
            for epoch in range(self.opts['numEpoch']):
                # loop over every data point

                ### TODO: YOUR CODE HERE
                # modify the code here so that the order of the data
                #   will be different each epoch
                # permute the data
                perm = np.arange(X.shape[0])
                util.permute(perm)
                X = X[perm, :]
                Y = Y[perm]
                # loop over every data point
                for n in range(X.shape[0]):
                    # supply the example to the online learner
                    self.nextExample(X[n], Y[n])

                # tell the online learner that we're
                # done with this iteration
                self.nextIteration()
        else:
            util.raiseNotDefined()


class AveragedPerceptron(Perceptron):

    def reset(self):
        """
        Reset the internal state of the classifier.
        """

        self.numUpd  = 0    # number of updates made
        self.weights = 0    # our weight vector
        self.bias    = 0    # our bias

        ### TODO: YOUR CODE HERE
        self.u       = 0    # cached weights
        self.B       = 0    # chached bias
        self.c       = 1# counter

    def nextExample(self, X, Y):
        """
        X is a vector training example and Y is its associated class.
        We're guaranteed that Y is either +1 or -1.  We should update
        our weight vector and bias according to the perceptron rule.
        """

        # check to see if we've made an error
        if Y * self.predict(X) <= 0:   ### SOLUTION-AFTER-IF
            self.numUpd = self.numUpd + 1

            ### TODO: YOUR CODE HERE
            # perform an update
            self.weights = self.weights + Y*X    # update weights
            self.bias    = self.bias + Y    # update bias
            self.u       = self.u + Y*X*self.c    # update chached wieghts
            self.B       = self.B + Y*self.c    # update chached bias

        ### TODO: YOUR CODE HERE
        self.c = self.c+1              # increment counter

    def train(self, X, Y):
        """
        (BATCH ONLY)

        X is a matrix of data points, Y is a vector of +1/-1 classes.
        """
        if self.online():
            for epoch in range(self.opts['numEpoch']):
                # loop over every data point
                for n in range(X.shape[0]):
                    # supply the example to the online learner
                    self.nextExample(X[n], Y[n])

                # tell the online learner that we're
                # done with this iteration
                self.nextIteration()

            ### TODO: YOUR CODE HERE
            self.weights = self.weights - (self.u/self.c)   # return averaged weights
            self.bias    = self.bias - (self.B/self.c)  # return averaged bias
        else:
            util.raiseNotDefined()
