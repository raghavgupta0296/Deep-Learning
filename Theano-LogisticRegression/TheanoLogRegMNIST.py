import cPickle
import gzip
import theano
import theano.tensor as T
import numpy as np


class LogReg(object):
    def __init__(self,input,n_in,n_out):
        self.W = theano.shared(value=np.zeros((n_in,n_out),dtype=theano.config.floatX),name='W',borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,),dtype=theano.config.floatX),name='b',borrow=True)
        self.p_ygivenx = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.prediction = T.argmax(self.p_ygivenx,axis=1)
        self.params=[self.W,self.b]
        self.input = input

    def negLogLike(self,y):
        return -T.mean(T.log(self.p_ygivenx)[T.arange(y.shape[0]),y])

    def errors(self,y):
        if y.ndim != self.prediction.ndim:
            raise TypeError('y should have the same shape as self.y_pred',('y', y.type, 'y_pred', self.prediction.type))

        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.prediction,y))
        else:
            raise NotImplementedError()


f1 = gzip.open('mnist.pkl.gz','rb')
trainset,validset,testset = cPickle.load(f1)

f1.close()

def shareddataset(dataxy):
    datax,datay = dataxy
    sharedx = theano.shared(np.asarray(datax, dtype=theano.config.floatX), borrow=True)
    sharedy = theano.shared(np.asarray(datay, dtype=theano.config.floatX), borrow=True)
    sharedy = sharedy.flatten()
    return sharedx,T.cast(sharedy,'int32')

# validset = testset
trainX,trainY = shareddataset(trainset)
testX,testY = shareddataset(testset)
validX,validY = shareddataset(validset)

alpha = 0.13
maxEpochs = 1000
batch_size = 600

ntrain = trainX.get_value(borrow=True).shape[0]/batch_size
ntest = testX.get_value(borrow=True).shape[0]/batch_size
nvalidate = validX.get_value(borrow=True).shape[0]/batch_size

index = T.lscalar()
x = T.matrix('x')
y = T.ivector('y')

classifier = LogReg(input=x,n_in=28*28,n_out=10)

cost = classifier.negLogLike(y)

g_w = T.grad(cost,classifier.W)
g_b = T.grad(cost,classifier.b)

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=[(classifier.W,classifier.W-alpha*g_w),(classifier.b,classifier.b-alpha*g_b)],
    givens={
        x: trainX[index*batch_size:(index+1)*batch_size],
        y: trainY[index*batch_size:(index+1)*batch_size]
    }
)

test_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={x: testX[index*batch_size:(index+1)*batch_size],
            y: testY[index*batch_size:(index+1)*batch_size]}
)

validation_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={x: testX[index*batch_size:(index+1)*batch_size],
            y: testY[index*batch_size:(index+1)*batch_size]}
)


epochs=0
stopCondition = False
patience = 5000
patience_inc = 2
validation_freq = min(ntrain,patience/2)
best_validationLoss = np.inf
improvement_threshold = 0.995
while epochs<maxEpochs and (not stopCondition):
    epochs += 1
    for miniIndex in xrange(ntrain):
        miniCost = train_model(miniIndex)
        iter = (epochs-1)*ntrain+miniIndex
        if iter%validation_freq == 0:
            validMiniLosses = [validation_model(i) for i in xrange(nvalidate)]
            validationLoss = np.mean(validMiniLosses)
            print "Epoch",epochs," minibatch",miniIndex+1,"/",ntrain,"  validation loss",validationLoss

            if validationLoss < best_validationLoss:
                if validationLoss < best_validationLoss * improvement_threshold:
                    patience = max(patience,iter*patience_inc)
                best_validationLoss = validationLoss

                testMiniLoss = [test_model(i) for i in xrange(ntest)]
                testLoss = np.mean(testMiniLoss)
                print "Epoch", epochs, " minibatch", miniIndex + 1,"/",ntrain, "  Test loss", testLoss

                with open('best_model.pkl','w') as f:
                    cPickle.dump(classifier,f)

        if patience<iter:
            stopCondition = True
            break

print " // Best Validation Score : ",best_validationLoss, " Test Performance : ",testLoss*100



















