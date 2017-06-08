# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:05:10 2017

@author: song-isong-i
"""


from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import numpy as np
import unicodecsv as csv
import pylab
import math

def main():
    X_train,Y_train,X_test,Y_test = load_data('sconcepts.csv','newlabels.csv')
    clf = SVR(kernel='rbf',gamma = 0.005, C=1.5)
    print(clf)
    print(X_train.shape, Y_train.shape)
    Estimator = clf.fit(X_train,Y_train)
    predicted = Estimator.predict(X_test)
    print(predicted[:5])
    print(Y_test[:5])
    print(clf.score(X_test,Y_test))
    plot(predicted, Y_test)
 
    
def load_data(Xfile,Yfile):
    X = np.loadtxt(Xfile,delimiter=',', dtype=np.float32)
    Y = np.loadtxt(Yfile,delimiter=',', dtype=np.float32)
    
    num_training = 13000
    X_train = X[:num_training]
    Y_train = Y[:num_training]
    X_test = X[num_training:]
    Y_test = Y[num_training:]
    return X_train, Y_train, X_test, Y_test
    
def plot(predicted, Y_test):

    pylab.xlabel('data')
    pylab.ylabel('predicted values\nlabels')
    #data = range(len(Y_test))
    #pylab.xlim(0,len(data))
    #pylab.ylim(0,0.2)
    #pylab.plot(data, likes, label = 'likes')
    #pylab.plot(data, Y_test, label = 'labels')
    pylab.plot(Y_test, predicted, 'go')
    #pylab.plot(data, labels, label = 'labels(likes/followers)')
    pylab.legend()
    pylab.show()
    

def test_error(v1,v2):
    e = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            e += 1
    er = (e / len(v1)) * 100
    return er
    
def cross_val_error(v):
    s = 0
    for n in v:
        s += n
    return 100 - (s/len(v))*100
    
    
    
    
#main()
    
