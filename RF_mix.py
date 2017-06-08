# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:31:32 2017

@author: song-isong-i

Random Forest mix

"""


from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error

import csv

def file_to_list(file):
    data = []
    print(file)
    f = open(file, 'rU')
    contents = csv.reader(f.read().splitlines())
    count = 0
    try:
        for c in contents:
            count += 1
            data.append(c)
    except Exception as e:
        print("count",count)
        raise e

    return data

def important_lists(data):
    likes = []
    followers = []
    count = 0
    try:
        for d in data:
            count += 1
            likes.append(int(d[1]))
            followers.append(int(d[7]))
    except Exception as e:
        print("count", count)
        raise e
    return likes, followers

data = file_to_list('posts_filtered.csv')
likes, followers = important_lists(data)
concepts = np.loadtxt('sconcepts.csv',delimiter=',', dtype=np.float32)
Y = np.array(likes[:])

likes = np.array(likes)
followers = np.array(followers)
followers = followers.reshape(1500,10)
likes = likes.reshape(1500,10)
for i in range(len(likes)):
    likes[i] = np.repeat(np.mean(likes[i]),10)
likes = likes.reshape(15000,)
followers = followers.reshape(15000,)
mY = np.mean(Y)
sdY = np.std(Y)
likes = (likes - np.mean(likes)) / np.std(likes)
followers = (followers - np.mean(followers)) / np.std(followers)
concepts = (concepts - np.mean(concepts)) / np.std(concepts)
Y = (Y-np.mean(Y))/np.std(Y)

X = []
for i in range(len(concepts)):
    x=np.hstack((concepts[i],likes[i]))
    x=np.hstack((x,followers[i]))
    X.append(x)
X = np.array(X)
print(X.shape)
print(X)


num_training = 12500
X_train = X[:num_training]
Y_train = Y[:num_training]
X_test = X[num_training:]
Y_test = Y[num_training:]


regressor = RandomForestRegressor(n_estimators=150)
regressor.fit(X_train, Y_train)
y = regressor.predict(X_test)
print(regressor.score(X_test,Y_test))
print('predicted :',y[:5])
print("label :",Y_test[:5])
print("AVERAGE DIFFERENCE B/W PREDICTED VALUE & ACTUAL LABEL:")
print( sum( [abs( y[i] - Y_test[i]) for i in range(len(y)) ] ) / len(y) )

test = [1,100,500,2364,9583,12345,14987,14999]
for t in test:
    pv = regressor.predict(X[t])
    pv = pv * sdY + mY
    print("predicted :",pv," // real :", Y[t]*sdY + mY)

import pylab
pylab.plot(Y_test, y, 'go')
pylab.legend()
pylab.show()

