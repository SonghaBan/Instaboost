# -*- coding: utf-8 -*-
"""
Created on Sun May 21 00:29:19 2017

@author: song-isong-i

Gradient Boosting Regressor
"""

from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
import pylab

import csv

def main():
    X,Y,mlikes,sdlikes = loaddata()
    X_train,Y_train,X_test,Y_test = splitdata(X,Y)
    est,y = GBR(X_train,Y_train,X_test,Y_test)
    plot(est,y)
    
    
def loaddata():    
    data = file_to_list('posts_filtered.csv')
    likes, followers = important_lists(data)
    concepts = np.loadtxt('4096features.csv',delimiter=',', dtype=np.float32)
    Y = np.array(likes[:])
    likes = np.array(likes)
    followers = np.array(followers)
    followers = followers.reshape(1500,10)
    likes = likes.reshape(1500,10)
    for i in range(len(likes)):
        likes[i] = np.repeat(np.mean(likes[i]),10)
    likes = likes.reshape(15000,)
    followers = followers.reshape(15000,)
    mlikes = np.mean(likes)
    sdlikes = np.std(likes)
    likes = (likes - mlikes) / sdlikes
    followers = (followers - np.mean(followers)) / np.std(followers)
    concepts = (concepts - np.mean(concepts)) / np.std(concepts)
    #Y = (Y-np.mean(Y))/np.std(Y)
    X = []
    for i in range(len(concepts)):
        x=np.hstack((concepts[i],likes[i]))
        x=np.hstack((x,followers[i]))
        X.append(x)
    X = np.array(X)
    return X,Y,mlikes,sdlikes

def splitdata(X,Y):
    num_training = 12500
    X_train = X[:num_training]
    Y_train = Y[:num_training]
    X_test = X[num_training:]
    Y_test = Y[num_training:]
    return X_train,Y_train,X_test,Y_test
    
def GBR(X_train,Y_train,X_test,Y_test):
    est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                                max_depth=2, random_state=0, loss='ls').fit(X_train, Y_train)
    y = est.predict(X_test)
    e = mean_squared_error(Y_test, y)
    print("MSE :", e)
    print('predicted :',y[:10])
    #print('poly :',y)
    print("label :",Y_test[:10])
    print("AVERAGE DIFFERENCE B/W PREDICTED VALUE & ACTUAL LABEL:")
    print( sum( [abs( y[i] - Y_test[i]) for i in range(len(y)) ] ) / len(y) )
    return est,y
    
def predictor(x, est, mlikes,sdlikes):
    y = est.predict(x)
    return y * sdlikes + mlikes

def plot(predicted, Y_test):
    pylab.plot(Y_test, predicted, 'go')
    pylab.legend()
    pylab.show()
    
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

main()