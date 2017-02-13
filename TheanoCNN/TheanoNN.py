# Raghav Gupta
# AI/Mixed Reality Lab
# Next Tech Lab

# Neural Network class which creates a HidLayer object for each Hidden Layer, and an object of Output Layer
# The neural network performs softmax; Negative Log Likelihood for loss; and Gradient Descent Optimisation using Back Propogation

import theano.tensor as T
import numpy as np
import theano

class HidLayer(object):
    def __init__(self, input, n_in, n_out, rng):
        self.input = input
        weights = np.asarray(
            rng.uniform(low=-np.sqrt(6/(n_in+n_out)), high=np.sqrt(6/(n_in+n_out)), size=(n_in,n_out)),
            dtype=theano.config.floatX)
        bias = np.zeros((n_out,),dtype=theano.config.floatX)
        self.W = theano.shared(weights,'W',borrow=True)
        self.b = theano.shared(bias,'b',borrow=True)
        self.output = T.tanh((T.dot(input,self.W)+self.b))
        self.params = [self.W,self.b]

class OutputLayer(object):
    def __init__(self, input, n_in, n_out, rng):
        self.input = input
        weights = np.asarray(rng.uniform(size=(n_in,n_out)),dtype=theano.config.floatX)
        bias = np.zeros((n_out,),dtype=theano.config.floatX)
        self.W = theano.shared(weights,'W',borrow=True)
        self.b = theano.shared(bias,'b',borrow=True)
        self.output = T.nnet.softmax((T.dot(input,self.W)+self.b))
        self.prediction = T.argmax(self.output,axis=1)
        self.params = [self.W,self.b]

class NN(object):
    def __init__(self, input, rng, n_in, n_hidden1, n_out):
        self.hl1 = HidLayer(input,n_in,n_hidden1,rng)
        # self.hl2 = HidLayer(self.hl1.output,n_hidden1,n_hidden2,rng)
        self.outl = OutputLayer(self.hl1.output,n_hidden1,n_out, rng)

        self.L1 = abs(self.hl1.W).sum() + abs(self.outl.W).sum()
        self.L2 = abs(self.hl1.W**2).sum() + abs(self.outl.W**2).sum()

        self.params = self.hl1.params+ self.outl.params
        self.input = input

    def negLogLik(self,y):
        return -T.mean(T.log(self.outl.output)[T.arange(y.shape[0]),y])

    def error(self,y):
        return T.mean(T.neq(self.outl.prediction,y))





