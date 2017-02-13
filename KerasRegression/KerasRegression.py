import pandas
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

dataframe=pandas.read_csv("housingdata.csv",delim_whitespace=True,header=None)
dataset=dataframe.values

X = dataset[:,0:13]
Y = dataset[:,13]

def baseline_model():
    model=Sequential()
    model.add(Dense(13,input_dim=13,init='normal',activation='relu'))
    model.add(Dense(1,init='normal'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

seed=7
np.random.seed(seed)

estimator=KerasRegressor(build_fn=baseline_model,nb_epoch=100,batch_size=5,verbose=0)
kfold=KFold(n=len(X),n_folds=10,random_state=seed)
results=cross_val_score(estimator,X,Y,cv=kfold)

print ("Results : %.2f (%.2f) MSE"%(results.mean(),results.std()))