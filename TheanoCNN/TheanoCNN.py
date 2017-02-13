import theano.tensor as T
import numpy as np
from TheanoNN import NN
import theano
from CNNLayer import CNNLayerWithMaxPool
# import matplotlib.pyplot as plt
import cPickle, gzip

rng = np.random.RandomState(1234)

f = gzip.open("mnist.pkl.gz",'rb')
trainset, validset, testset = cPickle.load(f)

# z = trainset[0][0]
# z = np.array(z)
# z = z.reshape((28,28))
# plt.imshow(z)
# plt.show()

def shared_dataset(dataset):
    x,y = dataset
    x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
    y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
    return x,T.cast(y,'int32')

trainX,trainY = shared_dataset(trainset)
validX,validY = shared_dataset(validset)
testX,testY = shared_dataset(testset)

batch_size = 500
nmaps = [20,50]
index=T.lscalar()
alpha = 0.1

x = T.matrix('x')
y = T.ivector('y')

inputLayer0 = x.reshape((batch_size,1,28,28))

layer0 = CNNLayerWithMaxPool(
    rng=rng,
    input=inputLayer0,
    filter_shape=(nmaps[0],1,5,5),
    im_shape=(batch_size,1,28,28),
    pooling=(2,2)
)

layer1 = CNNLayerWithMaxPool(
    rng=rng,
    input=layer0.output,
    filter_shape=(nmaps[1],nmaps[0],5,5),
    im_shape=(batch_size,nmaps[0],12,12),
    pooling=(2,2)
)

inputLayer2 = layer1.output.flatten(2)

layer2 = NN(inputLayer2,rng,nmaps[1]*4*4,500,10)

test_model = theano.function(
    inputs=[index],
    outputs=layer2.error(y),
    givens={
        x: testX[index*batch_size:(index+1)*batch_size],
        y: testY[index*batch_size:(index+1)*batch_size]
    }
)

valid_model = theano.function(
    inputs=[index],
    outputs=layer2.error(y),
    givens={
        x: validX[index * batch_size:(index + 1) * batch_size],
        y: validY[index * batch_size:(index + 1) * batch_size]
    }
)

params = layer2.params+layer1.params+layer0.params

cost = layer2.negLogLik(y)

grads = T.grad(cost,params)

train_model = theano.function(
    inputs=[index],
    outputs=cost,
    updates=[(param,param-alpha*grad) for param,grad in zip(params,grads)],
    givens={
        x: trainX[index * batch_size:(index + 1) * batch_size],
        y: trainY[index * batch_size:(index + 1) * batch_size]
    }
)

ntrain = trainX.get_value(borrow=True).shape[0]/batch_size
ntest = testX.get_value(borrow=True).shape[0]/batch_size
nvalidate = validX.get_value(borrow=True).shape[0]/batch_size

for i in range(100):
    print 'iteration : ', i
    for miniindex in xrange(ntrain):
        trainCost = train_model(miniindex)
        validLoss = [valid_model(p) for p in xrange(nvalidate)]
        validLoss = np.mean(validLoss)
        testLoss = [test_model(q) for q in xrange(ntest)]
        testLoss = np.mean(testLoss)
    print 'validLoss : ', validLoss
    print 'testLoss : ', testLoss

print 'Loss on valid set : ',validLoss
print 'Loss on test set : ',testLoss
