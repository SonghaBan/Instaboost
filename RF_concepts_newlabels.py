# -*- coding: utf-8 -*-
"""
Created on Sun May 21 19:30:26 2017

@author: song-isong-i

Random Forest / concepts, new label
"""

import scipy.stats as st
import numpy as np
import pylab
import csv
from sklearn.ensemble import RandomForestRegressor

def main():
    X, Y= load_data('sconcepts.csv','newlabels.csv')
    y = randomforest(X,Y)
    nY = relabel(y)
    plot(nY,1)
    

def load_data(Xfile,Yfile):
    X = np.loadtxt('sconcepts.csv',delimiter=',', dtype=np.float32)
    Y = np.loadtxt('newlabels.csv',delimiter=',', dtype=np.float32)
    return X,Y

def randomforest(X,Y):
    regressor = RandomForestRegressor(n_estimators=200)
    regressor.fit(X,Y)
    y = regressor.predict(X)
    print(regressor.score(X,Y))
    print('predicted :',y)
    print("label :",Y)
    return y
    
def relabel(y):
    mean = np.mean(y)
    sd = np.std(y)
    nd = st.norm(mean,sd)
    print("relabelling..")
    lo = open('img-likes.csv',"w")
    writer = csv.writer(lo, lineterminator='\n')
    for i in range(len(y)):
        y[i] = nd.cdf(y[i])
        writer.writerow([y[i]])            
    return y

def plot(Y, c):
    data = range(len(Y))
    pylab.figure(c)
    pylab.plot(data,Y)
    pylab.legend()
    pylab.show()
    
main()