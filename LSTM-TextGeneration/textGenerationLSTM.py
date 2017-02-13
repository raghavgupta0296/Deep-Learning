import numpy

from keras.models import Sequential,model_from_json
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,Callback

f = open("AliceInWonderland.txt")
data = f.read()
data = data.lower()
# print data

chars = set(data)
chars = sorted(chars)
# print chars

char2int = dict((c,i) for i,c in enumerate(chars))
# print char2int

nChars = len(chars)
nData = len(data)
# print nChars,nData

lettersToRead = 100
X=[]
Y=[]
for i in range(0,nData-lettersToRead,1):
    frame = data[i:i+lettersToRead]
    output = data[i+lettersToRead]
    X.append([char2int[f] for f in frame])
    Y.append([char2int[o] for o in output])
# X = X[400:660]
# Y = Y[400:660]
nX = len(X)
# print nX

dataX = X
X = numpy.reshape(X,(nX,lettersToRead,1))
X = X/float(nChars)
Y = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(256,input_shape=(X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1],activation="softmax"))
model.compile(loss='categorical_crossentropy',optimizer='adam')

f2 = "weights-improvement.hdf5"
checkpoint = ModelCheckpoint(f2,monitor='loss',verbose=1,save_best_only=True,mode='min')

g=open("generateTexts.txt","w")
g.close()

class predHist(Callback):
    def on_epoch_end(self,epoch,logs={}):
        self.generateText()

    def generateText(self):
        f2 = "weights-improvement.hdf5"
        model.load_weights(f2)
        model.compile(loss='categorical_crossentropy',optimizer='adam')
        int2char = dict((i,c) for i,c in enumerate(chars))
        start = numpy.random.randint(0,len(dataX)-1)
        pattern = dataX[start]
        datatowrite = [int2char[value] for value in pattern]
        g = open("generateTexts.txt", "a")
        g.write("\n")
        g.write(str("".join(datatowrite)))
        print "/","".join([int2char[value] for value in pattern]),"/",
        print " pattern length : ",len(pattern)
        for i in range(1000):
            # for debugging
            if len(pattern) != lettersToRead:
                pattern = pattern[:lettersToRead]

            x = numpy.reshape(pattern,(1,len(pattern),1))
            x = x/float(nChars)
            prediction = model.predict(x,verbose=0)
            prediction = numpy.argmax(prediction)
            prediction = int2char[prediction]
            print " Prediction : ", prediction,
            pattern.append(char2int[prediction])
            g.write(prediction.rstrip("\n"))
            pattern = pattern[1:len(pattern)]
        g.close()
p = predHist()
callback_list = [checkpoint,p]

model.fit(X,Y,nb_epoch=20,batch_size=128,callbacks=callback_list)

# model_json = model.to_json()
# with open("model.json","w") as json_file:
#     json_file.write(model_json)
# model.save_weights("model.h5")

# predHist.generateText()

