#Package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import itertools
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from ase import Atoms
from ase.io import read, write
from ase.visualize import view
from ase.visualize.plot import plot_atoms

#This class handle model/data imports
class Model_Importer():
    def __init__(self,data_path,model_save_path,feature_count):
        self.data_path = data_path #(str): stored data path/filename
        self.model_save_path = model_save_path #(str): stored model path
        self.features = feature_count
    def import_model(self):
        '''
        outputs:
        model (torch sequential object) - MLP NN model
        '''
        model = Sequential([
                            Dense(64,input_shape=(self.features,),activation='relu', use_bias=False),
                            Dense(64,activation='relu',use_bias=False),
                            Dense(12,activation='relu',use_bias=False),
                            Dense(1,activation="linear",use_bias=False),])
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mse', metrics=['mae','mse'])
        model.load_weights(self.model_save_path)
        return model

    #Import Train/val/test split of pre-processed data
    def import_data(self):
        with open(self.data_path, 'rb') as f:
            X_train = np.load(f)
            X_val = np.load(f)
            X_test = np.load(f)
            Y_train = np.load(f)
            Y_val = np.load(f)
            Y_test = np.load(f)
        return X_train,X_val,X_test,Y_train,Y_val,Y_test
    
    #Print model MAE, parity plot and option to save img PNG
    def model_performance(self,model,X_train,X_test,Y_train,Y_test ,save_img=False): 
        '''
        inputs:
        X/Y - array
        save_img - bool
        
        outputs:
        mean_absolute_error (float) - MAE of test set
        '''
        
        #Predicted Values
        yhat_train =list( model.predict(X_train)[:,0] )
        yhat_test  =list( model.predict(X_test)[:,0]  ) 
        preds=yhat_test+yhat_train

        #Real Values
        y_test= list(Y_test[:,0])
        y_train=list(Y_train[:,0])
        real = y_test+y_train
        
        labels = ['test' for _ in yhat_test] + ['train' for _ in yhat_train]
        #df of all datapoints
        d = {'Predicted': preds, 'Real': real,'Label':labels}
        df_scatter = pd.DataFrame(data=d)

        #MAE Values
        print(f'MAE train:  {str(mean_absolute_error(y_train, yhat_train))[:6]} eV')
        print(f'MAE test :  {str(mean_absolute_error(y_test, yhat_test))[:6]} eV')

        #Calculated std dev
        std_dev = np.std ( df_scatter['Predicted'] - df_scatter['Real'] )

        x = [-8,4]
        y = [-8-std_dev,4-std_dev]
        hue=['low','low']
        df1 = pd.DataFrame(data={'x':x,'lines': y,'hue':hue})

        x = [-8,4]
        y = [-8+std_dev,4+std_dev]
        hue=['up','up']
        df2 = pd.DataFrame(data={'x':x,'lines': y,'hue':hue})

        #Plot using seaborn scatterplot
        x4=sns.scatterplot(data=df_scatter, x='Predicted',y='Real',hue='Label',marker='o', color='b')
        x4.set(xlim=(-5.5,-1))
        x4.set(ylim=(-5.5,-1))
        x4.plot([-8,4],[-8,4],color='black',alpha=0.5) 
        x4=sns.lineplot(x='x',y='lines',data=df1,color='black',alpha  = 0.5)
        x4.lines[1].set_linestyle("--")
        x4=sns.lineplot(x='x',y='lines',data=df2,color='black',alpha  = 0.5)
        x4.lines[2].set_linestyle("--")
        x4.set_xlabel("Predicted Eads (eV)", fontsize = 15)
        x4.set_ylabel("DFT Eads (eV)", fontsize = 15)
        x4.set_title("Parity Plot (with 1 Standard Deviation Range)", fontsize = 20)
        plt.tight_layout()
        if save_img:
            plt.savefig('NN_Parity_Plot')
        return mean_absolute_error(y_test, yhat_test)
    
#This class handle optimization, exploration and generation of geometry files    
class Strucutre_Generator():
    def __init__(self,dictionnary,model,feature_count,nearest_atoms_considered):
        self.dictionnary = dictionnary
        self.model = model
        self.features = feature_count
        self.atom_count = nearest_atoms_considered
    #Called within Random_datapoint function. Perform feature embedding from a list of str values of 
    #local configuration into a tensor
    def convert_line(self, line):
        '''
        inputs:
        line (df series) - a pandas row from a dataframe of the dataset
        
        outputs:
        df_2D (df) - feature embedded datapoint
        '''
 
        label_holders = [str(i) for i in range(self.features)]
        readable_input = [self.dictionnary[i] for i in line]

        df_2D = pd.DataFrame(data=np.reshape(np.array(readable_input),(1,self.features)),columns=label_holders)
        return df_2D
        
    #Called within Generate_structures_2n. Generate random datapoint, convert to tensor    
    def random_datapoint(self, ele):
        '''
        inputs:
        ele (str) - element to alloy Cu with
        
        outputs:
        template (list) - atomic  configuration
        converted (df) - embedded atomic configuration
        '''
        
        template = ['Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu']
   
        replacements = [k for k in range(self.atom_count) if random.randrange(4) == 0]
        for i in replacements:
            template[i] = ele

        converted=self.convert_line(template)
        return template,converted
    
    #Generate fixed amount of random structures and save the structures/predictions within a list   
    def generate_structures_2n(self,pred_ele,count=100,save=False):
        '''
        inputs:
        pred_ele (str) - element to alloy Cu with 
        count (float) - number of generated  structures 
        save (bool) - self explanatory
        
        outputs:
        predictions_str (list of str) - predicted values in str format
        prediction (list of float) - predicted value in float format
        struct_pred (list or str) - generated structures
        '''
        
        t1=time.time()        
        predictions_str = []
        predictions=[]
        struct_pred = []
        print(f'Generating {pred_ele} BACs...')   
        
        #list comprehension would make this code hard to read so I left as is.
        for i in range(count): 
            random_template,datapoint = self.random_datapoint(pred_ele)
            prediction = self.model.predict(datapoint)

            predictions_str.append(str(prediction[0][0]))
            predictions.append(prediction[0][0])
            struct_pred.append(random_template)
        t2=time.time()      
        print(f'2n Structure Gen. Runtime (s): {t2-t1}')

        plt.hist(predictions,bins=25)
        plt.title(f'{count} structure predictions with {pred_ele} binary alloy')
        plt.xlabel('Eads (eV)')
        plt.ylabel('Frequency')
        
        if save:
            json_structures = f"structures_{pred_ele}.json"
            with open(json_structures, 'w') as f:
                json.dump(struct_pred, f)
                
            json_Eads = f"Eads_{pred_ele}.json"
            with open(json_Eads, 'w') as f:
                json.dump(predictions_str, f)
                
        self.predictions_2n = predictions
        self.predictions_structures_2n = struct_pred
        self.alloyed_element_2n = pred_ele
        
        return predictions_str,predictions,struct_pred
    
    def get_generation_stats(self,preds):
        '''
        inputs:
        preds (list) - list of predicted values
        '''
        
        preds = np.array(preds)
        print(f'Mean: {np.mean(preds):.5f}')
        print(f'Range: {np.max(preds) - np.min(preds):.5f}')
        print(f'Best Ads: {np.max(preds):.5f}')

    def get_optimal_structure(self,preds,structs,elements):
        '''
        inputs:
        preds (list) - list of predicted values
        structs (list) - list of predicted structures
        elements (list of str) - element to extract idx of
        
        outputs:
        structs (list) - optimal structures
        replacement_idx (list) - index of replacement atoms
        '''
        
        for i in range(len(preds)):
            if preds[i] == np.min(preds):
                idx=i
        
        replacement_idx = [i + 1 for i in range(len(structs[idx])) if structs[idx][i] in elements]
        
        return structs[idx],replacement_idx
    
    #Convert tensor into a geometry file for DFT calcs
    def new_design_2n(self,replacements,element,slab=None,save=False,formatting='espresso-in',mapping_file='Cu_pure_mapping.json'):
        '''
        inputs:
        replacements (list) - indexes to replace
        element (str) - replacement elements
        slab (ASE atoms object) - structure to edit
        save (bool) - self explanatory
        formating (str of ASE formats) - save file type
        mapping_file (str) - structure index mapping of atomic positions

        outputs:
        sample_atom (ASE atoms object) - optimized structure
        '''

        with open(mapping_file, "r") as json_file:
            atoms_map = json.load(json_file)

        if slab is None:
            sample_atom = read('Cu_Pure',format='vasp')
        if slab is not None:
            sample_atom = slab

        atom_list = sample_atom.get_chemical_symbols()
        for i in replacements:
            atom_list[atoms_map[str(i)]] = element

        sample_atom.set_chemical_symbols(atom_list)

        if save:
            file_name = f'{element}.relax'
            write(file_name,atm2,format=formatting )   
        return sample_atom
        
    def random_datapoint_3n(self, ele1,ele2):
        '''
        inputs:
        ele1/ele2 (str) - elements to alloy with Cu
        
        outputs:
        template (list) - atomic  configuration
        converted (df) - embedded atomic configuration
        '''
        
        template = ['Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu']
        
        replacements = [k for k in range(13) if random.randrange(3) == 0]
        
        ele1_replace = [j for j in replacements if random.randrange(2) == 0]
        ele2_replace = [j for j in replacements if random.randrange(2) == 1]

        for i in ele1_replace:
            template[i] = ele1

        for i in ele2_replace:
            template[i] = ele2
            
        converted=self.convert_line(np.array(template))
        return template,converted

    #Generate fixed amount of random structures and save the structures/predictions within a list       
    def generate_structures_3n(self,e1,e2,count=100,save=False):
        '''
        inputs:
        e1/e2 (str) - elements to alloy with Cu
        count (float) - number of generated  structures 
        save (bool) - self explanatory 
        
        outputs:
        predictions_str (list of str) - predicted values in str format
        prediction (list of float) - predicted value in float format
        struct_pred (list or str) - generated structures
        '''
        
        t1=time.time()

        predictions_str=[]
        predictions = []
        struct_pred = [] #list of lists

        print(f'Generating {e1}/{e2} TACs...')
        
        #list comprehension would make this code hard to read so I left as is.
        for i in range(count):
            template,datapoint = self.random_datapoint_3n(e1,e2)

            predictions_str.append(str(self.model.predict(datapoint)[0][0]))
            predictions.append(self.model.predict(datapoint)[0][0])
            struct_pred.append(template)
        t2=time.time()
        print(f'3n Structure Gen. Runtime (s): {t2-t1}')

        if save:
            json_structures = f"structures_{e1}_{e2}.json"
            with open(json_structures, 'w') as f:
                json.dump(struct_pred, f)

            json_Eads = f"Eads_{e1}_{e2}.json"
            with open(json_Eads, 'w') as f:
                json.dump(predictions_str, f)

        self.predictions_3n = predictions
        self.predictions_structures_3n = struct_pred
        self.alloyed_element_3n = e1+e2
        return predictions_str,predictions,struct_pred        
    
    #Convert list of str of elements into replace idx for new_design function 
    def symbol_to_index(self,ele,structure): 
        '''
        inputs:
        ele (str) - element to extract
        structure (list) - structure to convert
        
        outputs:
        idx (list) - converted structure of index
        '''
       
        idx = [i + 1 for i in range(len(structure)) if structure[i] == ele]
        
        return idx        

    #Convert tensor into a geometry file for DFT calcs         
    def new_design_3n(self,e1,e2,structure,save=False,formatting='espresso-in'):
        '''
        inputs:
        e1/e2 (str) - elements to alloy
        structure (list) - structure to convert          
        save (bool) - self explanatory
        formating (str) - extracted structure file format

        outputs:
        atm1/atm2 (ASE atoms object) - generated structures of BAC and TAC
        '''

        rep1 = self.symbol_to_index(e1,structure)
        rep2 = self.symbol_to_index(e2,structure)

        atm1 = self.new_design_2n(rep1,e1)
        atm2 = self.new_design_2n(rep2,e2,atm1)

        if save:
            file_name = f'{e1}_{e2}.relax'
            write(file_name,atm2,format=formatting )

        return atm1,atm2
        
    def visualize_slabs(self,optimal_BAC=None,optimal_TAC=None):
        '''
        inputs:
        optimal_BAC (ASE atoms object) - BAC to visualize
        optimal_TAC (ASE atoms object) - TAC to visualize          
        '''        
        if optimal_BAC is None:
            optimal_BAC = read('Cu_Pure',format='vasp')
        if optimal_TAC is None:
            optimal_TAC = read('Cu_Pure',format='vasp')
            
        fig, ax = plt.subplots(1,2)

        plot_atoms(optimal_BAC,ax[0],radii=0.9, rotation=('0x,0y,0z'))
        ax[0].set_title("BAC Optimized Structure")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        plot_atoms(optimal_TAC,ax[1],radii=0.9, rotation=('0x,0y,0z'))
        ax[1].set_title("TAC Optimized Structure")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()