# -*- coding: utf-8 -*-
"""
Created on Sun May 21 11:19:36 2017

@author: song-isong-i
"""

import scipy.stats as st
import numpy as np
import pylab
import csv
#
Y = np.loadtxt('labels.csv',delimiter=',', dtype=np.float32)
mean = np.mean(Y)
sd = np.std(Y)
nd = st.norm(mean, sd)

for i in range(len(Y)):
    
    Y[i] = nd.cdf(Y[i])
    
def relabel(label):
    print("relabelling..")
    with open('newlabels.csv', "w") as lo:
        writer = csv.writer(lo, lineterminator='\n')
        for l in label:        
            writer.writerow([l]) 

def plot(Y):
    Y.sort()
    data = range(len(Y))
    #pylab.xlim(0,len(data))
    #pylab.ylim(0,0.2)
    pylab.plot(data,Y)

    pylab.legend()
    pylab.show()


relabel(Y)