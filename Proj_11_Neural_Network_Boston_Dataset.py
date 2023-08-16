from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

boston = datasets.load_boston()
X = boston.data
y = boston.target 

ss = StandardScaler()
X = ss.fit_transform(X)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2)
model=Sequential()
model.add(Dense(15),input_dim=13,activation='relu')
model.add(Dense(1))
model.compile(loss='mean_squared_error')

history=model.fit(Xtrain,ytrain, epochs=150, batch_size=10)
ypred=model.predict(Xtest)
ypred=ypred[:,0]



rmse = (np.sqrt(mean_squared_error(ytest, ypred)))
r2 = r2_score(ytest, ypred)
print('Test RMSE =', rmse)
print('Test R2 score =', r2)