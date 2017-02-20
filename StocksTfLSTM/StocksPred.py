import pandas as pd
import numpy as np
import tensorflow as tf

# read csv
data = pd.read_csv("nvda.csv")
data = np.array(data["Close"])
# print (data)

# invert array ordering - old to new dates
data2 = []
for i in range(len(data)-1,-1,-1):
    data2.append(data[i])
data2 = np.array(data2)
data = data2
del data2
# print (data)

# normalize data
mina = min(data)
maxa = max(data)
data = (data-mina)/(maxa-mina)
# print (data)

lookback = 4

# make dataX:time-lookback-data and dataY:labels
def generate_data(data,lookback):
    dataX = []
    dataY = []
    for i in range(lookback,len(data),1):
        dataY.append(data[i])
        for j in range(lookback,0,-1):
            dataX.append(data[i-j])
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    dataX = np.reshape(dataX,[-1,lookback])
    dataY = np.reshape(dataY,[dataY.shape[0],1])
    return dataX,dataY

dataX, dataY = generate_data(data,lookback)
# print ((dataX),len(dataX))
# print ((dataY),len(dataY))

# split data into training-test sets
test_percentage = 0.90
test_size = int(len(dataX)*test_percentage)
trainX = dataX
trainY = dataY
testX = dataX[test_size:]
testY = dataY[test_size:]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# print ((testX,len(testX)))
# print ((trainX,len(trainX)))

x = tf.placeholder(tf.float32,[None,lookback,1])
y = tf.placeholder(tf.float32,[None,1])

# model
cell = tf.nn.rnn_cell.LSTMCell(lookback,state_is_tuple=True)
val, state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
val = tf.transpose(val,[1,0,2])
last = tf.gather(val,int(val.get_shape()[0])-1)
W = tf.Variable(tf.random_uniform([lookback,1],minval=2,maxval=4))
b = tf.Variable(tf.constant(0.1,shape=[1]))
output = tf.matmul(last,W)+b
output = tf.nn.relu(output)

# for un-normalizing data
unnormalize = output*(maxa-mina)+mina

# rms loss with adam optimizer
loss = tf.reduce_sum(tf.square(tf.sub(output,y)))
train_step = tf.train.RMSPropOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    for i in range(100):
        # print ("Epoch ",i)
        sess.run(init)
        train_step.run(feed_dict={x:trainX,y:trainY})
        if i%20==0:
            print ("Test Loss at epoch{} : ".format(i))
            acc = loss.eval(feed_dict={x:testX,y:testY})
            print (acc)
            print ("predicted : ",unnormalize.eval(feed_dict={x:trainX,y:trainY}))
            print ("real : ",np.array(testY*(maxa-mina)+mina))