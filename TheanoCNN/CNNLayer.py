import theano.tensor as T
import theano
from theano.tensor.nnet import conv
from theano.tensor.signal import pool
import numpy as np

class CNNLayerWithMaxPool(object):
    def __init__(self,rng,input,filter_shape,im_shape,pooling=(2,2)):
        self.input = input
        fin = filter_shape[1]*filter_shape[2]*filter_shape[3]
        fout = filter_shape[0]*filter_shape[2]*filter_shape[3]/(pooling[0]*pooling[1])

        self.W = theano.shared(
            name='W',
            value=np.asarray(
                rng.uniform(low=-np.sqrt(6/(fin+fout)),high=np.sqrt(6/(fin+fout)),size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True
        )
        self.b = theano.shared(
            name='b',
            value=np.zeros((filter_shape[0],),dtype=theano.config.floatX),
            borrow=True
        )
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            image_shape=im_shape,
            filter_shape=filter_shape
        )
        maxpool_out = pool.pool_2d(
            input=conv_out,
            ds=pooling,
            ignore_border=True
        )
        self.output = T.tanh(maxpool_out+self.b.dimshuffle('x',0,'x','x'))
        self.params = [self.W,self.b]



