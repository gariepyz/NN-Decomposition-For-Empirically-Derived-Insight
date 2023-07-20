import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.w1 = tf.keras.layers.Dense(6, activation='tanh', use_bias=False)
    self.w3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False)
  def call(self, inputs):
    x = self.w1(inputs)
    x = self.w3(x)
    return tf.math.reduce_sum(x, axis=1,)

class Model_analyzer:
    def __init__(self,descriptor_dict,position_count,dataframe):
        self.dictionnary = descriptor_dict
        self.position_count = position_count
        self.dataframe = dataframe
        
    def feature_embedding(self):
        #generate one hot encoded atomic positions (1-10)
        atomic_position_one_hot_encode=[]
        for i in range(self.position_count):
            position_i=[]
            for j in range(self.position_count):
                if i==j:
                    position_i.append(1)
                else:
                    position_i.append(0)
            atomic_position_one_hot_encode.append(position_i)

        #generate features to train model
        descriptor_df = np.empty((self.dataframe.shape[0],len(atomic_position_one_hot_encode[0]),len(atomic_position_one_hot_encode[0])+len(self.dictionnary[list(self.dictionnary.keys())[0]])))
        for k in range(self.dataframe.shape[0]):
            data_point = np.empty((descriptor_df.shape[1],descriptor_df.shape[2]))
            row=self.dataframe.iloc[k]
            for i in range(self.position_count):
                for j in range(self.position_count):
                    if i==j:
                        data_point[i,:] =  self.dictionnary[row[i]]+atomic_position_one_hot_encode[i]
            descriptor_df[k,:,:] = data_point
        return descriptor_df

    def NN_decomposition(self,site,model):
        #extract internal NN parameters and format a sample input
        params = model.trainable_variables
        p1 = np.array(params[0])
        p2 = np.array(params[1])

        site_conversion={}
        for i in range(self.position_count):
            position_i=[]
            for j in range(self.position_count):
                if i==j:
                    position_i.append(1)
                else:
                    position_i.append(0)
            site_conversion[i+1]=position_i

        elements=[]

        for i in list(self.dictionnary.keys()):
            elements.append(self.dictionnary[i]+site_conversion[site])
        single_element=np.reshape(np.array(elements),(1,5,13))

        #recreate basic NN from scratch using internal parameters
        step1 = np.dot(single_element,p1)
        step2 = tf.keras.activations.tanh(step1)
        step3 = np.dot(step2,p2)
        step4 = tf.keras.activations.linear(step3)
        step5 = tf.math.reduce_sum(step4, axis=1) #this step not needed because we don't want to pool individual contributions
        avg_influence = float(str(np.mean(abs(np.array(step4))))[:6])
        res = f'Average site {site} influence across all elements: {avg_influence} eV/atom'
        return avg_influence,res