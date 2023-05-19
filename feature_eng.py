#!/usr/bin/env python
# coding: utf-8

#Package imports
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

def feature_embedding(x):
    mapps = {'Co': [9,4,1.88],'Fe': [8,4,1.83],'Cu': [11,4,1.9],'Ni': [10,4,1.91],'Mo': [6,5,2.16]} 
    cols = list(x.columns)
    s0 = [1,0,0,0,0,0,0,0,0,0]
    s1 = [0,1,0,0,0,0,0,0,0,0]
    s2 = [0,0,1,0,0,0,0,0,0,0]
    s3 = [0,0,0,1,0,0,0,0,0,0]
    s4 = [0,0,0,0,1,0,0,0,0,0]
    s5 = [0,0,0,0,0,1,0,0,0,0]
    s6 = [0,0,0,0,0,0,1,0,0,0]
    s7 = [0,0,0,0,0,0,0,1,0,0]
    s8 = [0,0,0,0,0,0,0,0,1,0]
    s9 = [0,0,0,0,0,0,0,0,0,1]
    df_np = np.empty((x.shape[0],10,13))
    for j in range(x.shape[0]):
        l=np.empty((10,13))
        row = x.iloc[j]
        for i in range(10):
            if i==0:
                gs = mapps[row[i]]
                g = gs+s0
                l[i,:] = g
            if i==1:
                gs = mapps[row[i]]
                g = gs+s1
                l[i,:] = g
            if i==2:
                gs = mapps[row[i]]
                g = gs+s2
                l[i,:] = g            
            if i==3:
                gs = mapps[row[i]]
                g = gs+s3
                l[i,:] = g
            if i==4:
                gs = mapps[row[i]]
                g = gs+s4
                l[i,:] = g
            if i==5:
                gs = mapps[row[i]]
                g = gs+s5
                l[i,:] = g
            if i==6:
                gs = mapps[row[i]]
                g = gs+s6
                l[i,:] = g
            if i==7:
                gs = mapps[row[i]]
                g = gs+s7
                l[i,:] = g
            if i==8:
                gs = mapps[row[i]]
                g = gs+s8
                l[i,:] = g
            if i==9:
                gs = mapps[row[i]]
                g = gs+s9
                l[i,:] = g
        df_np[j,:,:] = l
    return df_np
def NN_decomposition(site,p1,p2):
    # site = s1,s2,s3,s4...s10 (specified site you want to study)
    site_conversion = {1:[1,0,0,0,0,0,0,0,0,0],
                       2:[0,1,0,0,0,0,0,0,0,0],
                       3:[0,0,1,0,0,0,0,0,0,0],
                       4:[0,0,0,1,0,0,0,0,0,0],
                       5:[0,0,0,0,1,0,0,0,0,0],
                       6:[0,0,0,0,0,1,0,0,0,0],
                       7:[0,0,0,0,0,0,1,0,0,0],
                       8:[0,0,0,0,0,0,0,1,0,0],
                       9:[0,0,0,0,0,0,0,0,1,0],
                       10:[0,0,0,0,0,0,0,0,0,1]}
    s = site_conversion[int(site)]
    Co=[9,4,1.88]+s #Co
    Fe=[8,4,1.83]+s #Fe
    Cu=[11,4,1.9]+s #Cu
    Ni=[10,4,1.91]+s #Ni
    Mo=[6,5,2.16]+s#Mo

    #sample input to gain chemical insights
    single_element= np.array([Co,Fe,Cu,Ni,Mo])
    single_element=np.reshape(single_element,(1,5,13))

    #recreate basic NN from scratch
    step1 = np.dot(single_element,p1)
    step2 = tf.keras.activations.tanh(step1)

    step3 = np.dot(step2,p2)
    step4 = tf.keras.activations.linear(step3)

    step5 = tf.math.reduce_sum(step4, axis=1)

    #output results
    ele_names=['Co','Fe','Cu','Ni','Mo']
    ele_cont = np.array(step4)
    
    avg_influence = np.mean(abs(ele_cont))
    print('Average site influence across all elements: '+str(avg_influence)[:6]+' eV/atom')
    
    return avg_influence