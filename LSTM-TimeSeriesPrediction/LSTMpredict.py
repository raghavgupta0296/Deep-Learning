import pandas
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.metrics import mean_squared_error
import math

dataset = pandas.read_csv("international-airline-passengers.csv",usecols=[1],skip_footer=3,engine='python')
# plt.plot(dataset)
# plt.show()

data = dataset.values
data = data.astype('float32')
# print data,type(data)
# print data.shape
# print data[143][0]

# Min-Max Scalar
a,b = min(data),max(data)
data = (data - a)/(b-a)
# print data

data = np.append(data,np.zeros([len(data),1]),1)
# print data

for i in range(0,len(data)-1):
    data[i][1] = data[i+1][0]
# print data

data = data[:-1][:]
# print data

ratio = 0.33
# print int((1-ratio)*len(data)),len(data)
trainData = data[:int((1-ratio)*len(data)),:]
# print trainData
testData = data[int((1-ratio)*len(data)):len(data),:]
# print testData

# LSTM
# print trainData
trainX,trainY = trainData[:,0],trainData[:,1]
# print trainX,trainY
testX,testY = testData[:,0],testData[:,1]

trainX = np.reshape(trainX,(trainX.shape[0],1,1))
testX = np.reshape(testX,(testX.shape[0],1,1))
# print trainX.shape

model = Sequential()
model.add(LSTM(4,input_dim=1))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX,trainY,nb_epoch=100,batch_size=1,verbose=2)

testPredict = model.predict(testX)
# print testPredict

testPredict = testPredict*(b-a)+a
testY = testY*(b-a)+a

# print testPredict

# print testY,testY.shape
# print testPredict,testPredict.shape

testScore = math.sqrt(mean_squared_error(testY,testPredict[:,0]))
print "test score : %0.2f RMSE"%testScore

print trainY.shape,testPredict.shape
predictPlot = np.empty_like(dataset)
predictPlot[:,:] = np.nan
# predictPlot[1:len(trainY)+1,:] = np.reshape(trainY,(trainY.shape[0],1))
predictPlot[len(trainY)+1:,:] = testPredict
plt.plot(dataset)
plt.plot(predictPlot)
plt.show()






