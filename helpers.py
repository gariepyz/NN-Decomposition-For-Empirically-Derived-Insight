import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.w1 = tf.keras.layers.Dense(6, activation='tanh', use_bias=False)
    self.w3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False)
  def call(self, inputs):
    x = self.w1(inputs)
    x = self.w3(x)
    return tf.math.reduce_sum(x, axis=1,)

class Model_Analyzer():
    def __init__(self,descriptor_dict,position_count,dataset_name):
        self.dictionnary = descriptor_dict
        self.position_count = position_count
        self.dataset_name = dataset_name
        
    def import_dataset(self,adsorbate):
        '''
        inputs:
        adsorbate (str) - adsorbate to analyze (CO/CHO/COOH) 
        
        outputs:
        df_ads (df) - datapoints of the specified adsorbate
        '''
        
        df = pd.read_csv(self.dataset_name)
        df_ads = df[df['Adsorbate']==adsorbate]       
        self.dataframe=df_ads       
        return df_ads    
    
    def feature_embedding(self):
        '''
        outputs:
        descriptor_df (df) - x inputs for NN training
        y (df) - y inputs for training
        '''
        
        #generate one hot encoded atomic positions (1-self.positions_count)
        #This could be simplified through list comprehension however it dramatically reduces reability
        atomic_position_one_hot_encode=[]
        for i in range(self.position_count):
            position_i=[]
            for j in range(self.position_count):
                if i==j:
                    position_i.append(1)
                else:
                    position_i.append(0)
            atomic_position_one_hot_encode.append(position_i)

        #generate features to train model using dictionnary
         #This could also be simplified through list comprehension however it dramatically reduces reability
        descriptor_df = np.empty((self.dataframe.shape[0],len(atomic_position_one_hot_encode[0]),len(atomic_position_one_hot_encode[0])+len(self.dictionnary[list(self.dictionnary.keys())[0]])))
        for k in range(self.dataframe.shape[0]):
            data_point = np.empty((descriptor_df.shape[1],descriptor_df.shape[2]))
            row=self.dataframe.iloc[k]
            for i in range(self.position_count):
                for j in range(self.position_count):
                    if i==j:
                        data_point[i,:] =  self.dictionnary[row[i]]+atomic_position_one_hot_encode[i]
            descriptor_df[k,:,:] = data_point
        y = self.dataframe['Eads']
        self.x = descriptor_df
        self.y = y
        return descriptor_df,y

    def train_model(self,split_size,epoch=3000):
        '''
        inputs:
        split_size (float) - train/test split
        epoch (int) - number of epochs to train for
        
        outputs:
        model (TF sequential) - trained model
        x/y (np.array) - train/test split data
        '''
        x_train, x_test, y_train, y_test = train_test_split(self.x,self.y, test_size=split_size)

        model = MyModel()
        model.compile(optimizer='adam', loss='mse', metrics=['mae','mse'])
        h = model.fit(x_train, y_train, epochs=epoch, callbacks=[],verbose=0 )
        
        self.model = model
        return model,x_train, x_test, y_train, y_test
    
    #NN decomposition alg for a single site
    def single_decomposition(self,site,model):
        '''
        inputs:
        site (int) - specific site to analyze
        model (TF sequantial) - model to use for predicttion
        
        outputs:
        avg_influence (float) - average influence from the specified site across all elements
        res (str) - string to print (optional)
        '''
        
        #extract internal NN parameters and format a sample input
        params = model.trainable_variables
        p1 = np.array(params[0])
        p2 = np.array(params[1])
        
        #This could be simplified through list comprehension however it dramatically reduces reability
        site_conversion={}
        for i in range(self.position_count):
            position_i=[]
            for j in range(self.position_count):
                if i==j:
                    position_i.append(1)
                else:
                    position_i.append(0)
            site_conversion[i+1]=position_i
        
        elements = [self.dictionnary[i] + site_conversion[site] for i in self.dictionnary.keys()]

        single_element=np.reshape(np.array(elements),(1,len(self.dictionnary),self.position_count +len(self.dictionnary[list(self.dictionnary.keys())[0]])))

        #recreate basic NN from scratch using internal parameters
        step1 = np.dot(single_element,p1)
        step2 = tf.keras.activations.tanh(step1)
        step3 = np.dot(step2,p2)
        step4 = tf.keras.activations.linear(step3)
        step5 = tf.math.reduce_sum(step4, axis=1) #this step not needed because we don't want to pool individual contributions
        
        avg_influence = np.mean(abs(np.array(step4)))
        res = f'Average site {site} influence across all elements: {avg_influence:.6f} eV/atom'

        return avg_influence,res
    
    def decompose(self,visualize=True):
        '''
        inputs:
        visualize (bool) - self explanatory
        
        outputs:
        sites (list) - sites of that were analyzed
        influences (list) - site influences (eV)
        '''
        
        sites = [i + 1 for i in range(self.position_count)]
        influences = [self.single_decomposition(site, self.model)[0] for site in sites]

        if visualize:        
            #Visualize Results
            plt.bar(sites, influences, align='center', alpha=1)
            plt.xlabel('Sites')
            plt.ylabel('Atomic Influence (eV/atom)')
            plt.title('Empirically Dervied Infuence of Local Chemical Environment')
            plt.xticks(sites)
            plt.grid(False)
            img = mpimg.imread('images/atomic_arrangement.png')
            plt.imshow(img, extent=[0, 11, 0, 1.5], aspect='auto', alpha=1)
        return sites, influences
    
    #used for the data visualization notebook
    def import_influences(self,ads):
        '''
        inputs:
        ads (str) - adsorbate of interest (COOH/CHO/CO)
        
        outputs:
        vis (df) - df of the adsorbate data
        '''
        
        vis = pd.read_csv('site_infs.csv')
        vis = vis[(vis['Adsorbate'] == ads) & (vis['Site'] == 1)]
        return vis
    
    #used for the data visualization notebook    
    def calculate_std(self,x_train, x_test, y_train, y_test):
        '''
        inputs:
        x/y (np.array) - training/test sets
        
        outputs:
        df (df) - predictions
        df1/2 (df) - std of predictions)
        '''
        
        label=[]

        test = [i[0] for i in self.model.predict(x_test).tolist()]
        [label.append('test') for l in test]

        train = [i[0] for i in self.model.predict(x_train).tolist()]
        [label.append('train') for l in train]

        pred_y = test + train
        real_y =list(y_test)+list(y_train)
        df = pd.DataFrame(data={'predict': pred_y, 'real': real_y,'label':label})
        std_dev= np.std(df['predict'] - df['real'])

        #Upper and low bounds for parity plot
        df1 = pd.DataFrame(data={'x':[-4,4],'lines': [-4-std_dev,4-std_dev],'hue':['low','low']})
        df2 = pd.DataFrame(data={'x':[-4,4],'lines': [-4+std_dev,4+std_dev],'hue':['up','up']})  
        return df,df1,df2