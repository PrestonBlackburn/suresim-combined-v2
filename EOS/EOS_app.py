from flask import request, jsonify, render_template, Blueprint
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
import math 
from flask_sqlalchemy import SQLAlchemy
import datetime

from VTPR import VTPR
from pandas import read_csv, DataFrame
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


EOS_app = Blueprint("EOS_app", __name__, static_folder = "static", template_folder = "templates")



## Class for LV input
class LV_Input:
    
    '''Converts LV_Input to LV, Mol, or W units
        Dependancies:
            Json
            pandas
            numpy
        Either directly enter: N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW into LV_Output
        or
        enter a dictionary from json file: LV_Output.from_dict(sample)
    '''
    
    Density = [6.727, 6.8129, 2.5, 2.9704 , 4.2285, 4.6925, 4.8706, 5.212, 5.2584, 5.5364]
    MW = [28.0, 44.0, 16.0, 30.1, 44.1, 58.1, 58.1, 72.1, 72.1, 86.2]
    comp_names =  ['Nitrogen', 'Carbon Dioxide', 'Methane',  
             'Ethane', 'Propane', 'Isobutane', 'n-butane', 
             'Isopentane', 'n-pentane', 'hexanes', 'heptanes+']
    
    
    def __init__(self, N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW):
        self.N2 = N2
        self.CO2 = CO2
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.iC4 = iC4
        self.C4 = C4
        self.iC5 = iC5
        self.C5 = C5
        self.C6 = C6
        self.C7 = C7
        self.shrink = shrink
        self.P = P
        self.T = T
        self.C7_SG = C7_SG
        self.SG = SG
        self.C7_MW = C7_MW
        
    def lv_pct(self):
        return (self.N2, self.CO2, self.C1, self.C2, 
                self.C3, self.iC4, self.C4, self.iC5,
                self.C5, self.C6, self.C7, self.shrink, 
                self.P, self.T, self.C7_SG, self.SG, self.C7_MW)
    
    
    def lv_to_w_calc(self):
        df_names = ['LV % N2', 'LV % CO2', 'LV % C1', 'LV % C2',
        'LV % C3', 'LV % IC4', 'LV % NC4', 'LV % IC5', 
        'LV % NC5', 'LV % C6', 'LV % C7', 'Shrinkage Factor',
        'Pressure', 'Temp', 'C7+ Specific Gravity', 'Total Spec. Grav.', 
         'C7+ Molecular Weight']
            
        df = pd.DataFrame([[self.N2, self.CO2, self.C1, self.C2, self.C3, 
                           self.iC4, self.C4, self.iC5,self.C5, self.C6, self.C7, self.shrink,
                          self.P, self.T, self.C7_SG, self.SG, self.C7_MW]], columns= df_names)

        LV_input_vals = df[['LV % N2', 'LV % CO2', 'LV % C1', 'LV % C2',
                    'LV % C3', 'LV % IC4', 'LV % NC4', 'LV % IC5', 
                    'LV % NC5', 'LV % C6', 'LV % C7']]
        LV_sum = np.sum(LV_input_vals, axis = 1)
        norm_LV = (LV_input_vals.values/LV_sum[:,None])*100
        norm_LV = pd.DataFrame(norm_LV, columns = self.comp_names)

        Density_df = norm_LV.iloc[:,0:10] * self.Density
        Density_df['heptanes+'] = norm_LV['heptanes+'] * df['C7+ Specific Gravity']*8.3372
        Density_sum = np.sum(Density_df, axis=1)
        W_pct = (Density_df/Density_sum[:,None])*100
        return W_pct, df
    
    
    def mol_pct(self):
        
        W_pct = self.lv_to_w_calc()[0]
        df = self.lv_to_w_calc()[1]

        MW_df = W_pct.iloc[:,0:10] / LV_Input.MW
        MW_df['heptanes+'] = W_pct['heptanes+'] / df['C7+ Molecular Weight']
        MW_sum = np.sum(MW_df, axis=1)
        mol_pct = (MW_df/MW_sum[:,None])*100
        mol_pct_vals = mol_pct.values
        
        self.N2 = mol_pct_vals[0,0]
        self.CO2 = mol_pct_vals[0,1]
        self.C1 = mol_pct_vals[0,2]
        self.C2 = mol_pct_vals[0,3]
        self.C3 = mol_pct_vals[0,4]
        self.iC4 = mol_pct_vals[0,5]
        self.C4 = mol_pct_vals[0,6]
        self.iC5 = mol_pct_vals[0,7]
        self.C5 = mol_pct_vals[0,8]
        self.C6 = mol_pct_vals[0,9]
        self.C7 = mol_pct_vals[0,10]
        
        return (self.N2, self.CO2, self.C1, self.C2, 
            self.C3, self.iC4, self.C4, self.iC5,
            self.C5, self.C6, self.C7, self.shrink, 
            self.P, self.T, self.C7_SG, self.SG, self.C7_MW)
    
    
    def w_pct(self):
        
        W_pct = self.lv_to_w_calc()[0]
        W_pct_vals = W_pct.values
        
        self.N2 = W_pct_vals[0,0]
        self.CO2 = W_pct_vals[0,1]
        self.C1 = W_pct_vals[0,2]
        self.C2 = W_pct_vals[0,3]
        self.C3 = W_pct_vals[0,4]
        self.iC4 = W_pct_vals[0,5]
        self.C4 = W_pct_vals[0,6]
        self.iC5 = W_pct_vals[0,7]
        self.C5 = W_pct_vals[0,8]
        self.C6 = W_pct_vals[0,9]
        self.C7 = W_pct_vals[0,10]
        
        return (self.N2, self.CO2, self.C1, self.C2, 
            self.C3, self.iC4, self.C4, self.iC5,
            self.C5, self.C6, self.C7, self.shrink, 
            self.P, self.T, self.C7_SG, self.SG, self.C7_MW)
     
        
    @classmethod
    def from_dict(cls, comp_dict):
        for sample_comp in comp_dict[list(comp_dict.keys())[0]]:
            N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW = (float(sample_comp['N2']),
                    float(sample_comp['CO2']), float(sample_comp['C1']), float(sample_comp['C2']),
                     float(sample_comp['C3']), float(sample_comp['iC4']), float(sample_comp['C4']), float(sample_comp['iC5']),
                     float(sample_comp['C5']), float(sample_comp['C6']), float(sample_comp['C7']), 1,
                     float(sample_comp['P']), float(sample_comp['T']), float(sample_comp['C7_SG']), float(sample_comp['SG']), 
                     float(sample_comp['C7_MW']))
        return cls(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW)
        

## new class for Mol% input

class Mol_Input:
    
    '''Converts Mol_Input to LV, Mol, or W units
        Dependancies:
            Json
            pandas
            numpy
        Either directly enter: N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW into LV_Output
        or
        enter a dictionary from json file: LV_Output.from_dict(sample)
    '''
    
    Density = [6.727, 6.8129, 2.5, 2.9704 , 4.2285, 4.6925, 4.8706, 5.212, 5.2584, 5.5364]
    MW = [28.0, 44.0, 16.0, 30.1, 44.1, 58.1, 58.1, 72.1, 72.1, 86.2]
    comp_names =  ['Nitrogen', 'Carbon Dioxide', 'Methane',  
             'Ethane', 'Propane', 'Isobutane', 'n-butane', 
             'Isopentane', 'n-pentane', 'hexanes', 'heptanes+']
    
    
    def __init__(self, N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW):
        self.N2 = N2
        self.CO2 = CO2
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.iC4 = iC4
        self.C4 = C4
        self.iC5 = iC5
        self.C5 = C5
        self.C6 = C6
        self.C7 = C7
        self.shrink = shrink
        self.P = P
        self.T = T
        self.C7_SG = C7_SG
        self.SG = SG
        self.C7_MW = C7_MW
        
    def mol_pct(self):
        return (self.N2, self.CO2, self.C1, self.C2, 
                self.C3, self.iC4, self.C4, self.iC5,
                self.C5, self.C6, self.C7, self.shrink, 
                self.P, self.T, self.C7_SG, self.SG, self.C7_MW)
    
    
    def mol_to_w_calc(self):
        df_names = ['mol % N2', 'mol % CO2', 'mol % C1', 'mol % C2',
        'mol % C3', 'mol % IC4', 'mol % NC4', 'mol % IC5', 
        'mol % NC5', 'mol % C6', 'mol % C7', 'Shrinkage Factor',
        'Pressure', 'Temp', 'C7+ Specific Gravity', 'Total Spec. Grav.', 
         'C7+ Molecular Weight']
            
        df = pd.DataFrame([[self.N2, self.CO2, self.C1, self.C2, self.C3, 
                           self.iC4, self.C4, self.iC5,self.C5, self.C6, self.C7, self.shrink,
                          self.P, self.T, self.C7_SG, self.SG, self.C7_MW]], columns= df_names)

        mol_input_vals = df[['mol % N2', 'mol % CO2', 'mol % C1', 'mol % C2',
                    'mol % C3', 'mol % IC4', 'mol % NC4', 'mol % IC5', 
                    'mol % NC5', 'mol % C6', 'mol % C7']]
        mol_sum = np.sum(mol_input_vals, axis = 1)
        norm_mol = (mol_input_vals.values/mol_sum[:,None])*100
        norm_mol = pd.DataFrame(norm_mol, columns = self.comp_names)

        MW_df = norm_mol.iloc[:,0:10] * Mol_Input.MW
        MW_df['heptanes+'] = norm_mol['heptanes+'] * df['C7+ Molecular Weight']
        MW_sum = np.sum(MW_df, axis=1)
        w_pct = (MW_df/MW_sum[:,None])*100
        return w_pct, df
     
    def lv_pct(self):
        W_pct = self.mol_to_w_calc()[0]
        df = self.mol_to_w_calc()[1]

        Density_df = W_pct.iloc[:,0:10] / LV_Input.Density
        Density_df['heptanes+'] = W_pct['heptanes+'] / (df['C7+ Specific Gravity'] * 8.3372)
        Density_sum = np.sum(Density_df, axis=1)
        lv_pct = (Density_df/Density_sum[:,None])*100
        lv_pct_vals = lv_pct.values
        
        self.N2 = lv_pct_vals[0,0]
        self.CO2 = lv_pct_vals[0,1]
        self.C1 = lv_pct_vals[0,2]
        self.C2 = lv_pct_vals[0,3]
        self.C3 = lv_pct_vals[0,4]
        self.iC4 = lv_pct_vals[0,5]
        self.C4 = lv_pct_vals[0,6]
        self.iC5 = lv_pct_vals[0,7]
        self.C5 = lv_pct_vals[0,8]
        self.C6 = lv_pct_vals[0,9]
        self.C7 = lv_pct_vals[0,10]
        
        return (self.N2, self.CO2, self.C1, self.C2, 
            self.C3, self.iC4, self.C4, self.iC5,
            self.C5, self.C6, self.C7, self.shrink, 
            self.P, self.T, self.C7_SG, self.SG, self.C7_MW)
     
    def w_pct(self):
        W_pct = self.mol_to_w_calc()[0]
        W_pct_vals = W_pct.values
        
        self.N2 = W_pct_vals[0,0]
        self.CO2 = W_pct_vals[0,1]
        self.C1 = W_pct_vals[0,2]
        self.C2 = W_pct_vals[0,3]
        self.C3 = W_pct_vals[0,4]
        self.iC4 = W_pct_vals[0,5]
        self.C4 = W_pct_vals[0,6]
        self.iC5 = W_pct_vals[0,7]
        self.C5 = W_pct_vals[0,8]
        self.C6 = W_pct_vals[0,9]
        self.C7 = W_pct_vals[0,10]
        
        return (self.N2, self.CO2, self.C1, self.C2, 
            self.C3, self.iC4, self.C4, self.iC5,
            self.C5, self.C6, self.C7, self.shrink, 
            self.P, self.T, self.C7_SG, self.SG, self.C7_MW)
     
        
    @classmethod
    def from_dict(cls, comp_dict):
        for sample_comp in comp_dict[list(comp_dict.keys())[0]]:
            N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW = (float(sample_comp['N2']),
                    float(sample_comp['CO2']), float(sample_comp['C1']), float(sample_comp['C2']),
                     float(sample_comp['C3']), float(sample_comp['iC4']), float(sample_comp['C4']), float(sample_comp['iC5']),
                     float(sample_comp['C5']), float(sample_comp['C6']), float(sample_comp['C7']), 1,
                     float(sample_comp['P']), float(sample_comp['T']), float(sample_comp['C7_SG']), float(sample_comp['SG']), 
                     float(sample_comp['C7_MW']))
        return cls(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW)
        
def n_stage(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW, Px, Tx):

    sample1 = Mol_Input(N2, C1, CO2, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW)

    mole_fraction = sample1.mol_pct()
    mole_fraction = mole_fraction[0:11]

    component_list = [
                'Nitrogen',
                'Methane',
                'Carbon Dioxide',
                'Ethane',
                'Propane',
                'Isobutane',
                'n-Butane',
                'Isopentane',
                'n-Pentane',
                'n-Hexane',
                'Heptanes+'
                ]

    MW_plus_fraction = float(C7_MW)
    #MW_plus_fraction = 181
    SG_plus_fraction = float(C7_SG)
    #SG_plus_fraction = 0.795


    T = (T - 32) * 5 / 9 + 273.15
    P = (P + 14.65)/14.5038
    Tx = (Tx - 32) * 5 / 9 + 273.15
    Px = (Px + 14.65)/14.5038

    eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
    shrinkage = eos.pt_flash(Px, Tx)

    # handeling components with a zero conc
    liquid = eos.x
    if component_list != list(liquid.keys()):
        missing_comp = [x for x in component_list if x not in liquid]
        zeros = []
        for i in range(len(missing_comp)):
            i = 0
            zeros.append(i)
        temp = dict(zip(missing_comp, zeros))
        liquid = liquid.to_dict()
        liquid.update(temp)

    else:
        pass

    liquid_df = pd.DataFrame(index=component_list)
    liquid_df["mol"] = pd.Series(liquid)
    liquid_df['mol pct'] = liquid_df['mol'] / liquid_df['mol'].sum() *100
    liquid_pct_df = liquid_df['mol pct']
    liquid_pct_df = pd.DataFrame(liquid_pct_df)


    # handeling components with a zero conc
    vapor = eos.y
    if component_list != list(vapor.keys()):
        missing_comp_v = [x for x in component_list if x not in vapor]
        zeros = []
        for i in range(len(missing_comp_v)):
            i = 0
            zeros.append(i)
        temp_v = dict(zip(missing_comp_v, zeros))
        vapor = vapor.to_dict()
        vapor.update(temp_v)
    else:
        pass

    vapor_df = pd.DataFrame(index=component_list)
    vapor_df["mol"] = pd.Series(vapor)
    vapor_df['mol pct'] = vapor_df['mol'] / vapor_df['mol'].sum() *100
    vapor_pct_df = vapor_df['mol pct']
    vapor_pct_df = pd.DataFrame(vapor_pct_df)


    N2 =liquid_pct_df.iloc[:,0][0]
    CO2 = liquid_pct_df.iloc[:,0][2]
    C1 = liquid_pct_df.iloc[:,0][1]
    C2 = liquid_pct_df.iloc[:,0][3]
    C3 = liquid_pct_df.iloc[:,0][4]
    iC4 =  liquid_pct_df.iloc[:,0][5]
    C4 =   liquid_pct_df.iloc[:,0][6]
    iC5 = liquid_pct_df.iloc[:,0][7]
    C5 = liquid_pct_df.iloc[:,0][8]
    C6 = liquid_pct_df.iloc[:,0][9]
    C7 = liquid_pct_df.iloc[:,0][10]

    return N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, liquid_pct_df, vapor_pct_df


def final_stage(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW):

    sample1 = Mol_Input(N2, C1, CO2, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW)

    mole_fraction = sample1.mol_pct()
    mole_fraction = mole_fraction[0:11]

    component_list = [
                'Nitrogen',
                'Methane',
                'Carbon Dioxide',
                'Ethane',
                'Propane',
                'Isobutane',
                'n-Butane',
                'Isopentane',
                'n-Pentane',
                'n-Hexane',
                'Heptanes+'
                ]

    MW_plus_fraction = float(C7_MW)
    #MW_plus_fraction = 181
    SG_plus_fraction = float(C7_SG)
    #SG_plus_fraction = 0.795


    T = (T - 32) * 5 / 9 + 273.15
    P = (P + 14.65)/14.5038

    eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
    shrinkage = eos.shrinkage(bubble_point_force=False)

    # handeling components with a zero conc
    liquid = eos.x
    if component_list != list(liquid.keys()):
        missing_comp = [x for x in component_list if x not in liquid]
        zeros = []
        for i in range(len(missing_comp)):
            i = 0
            zeros.append(i)
        temp = dict(zip(missing_comp, zeros))
        liquid = liquid.to_dict()
        liquid.update(temp)

    else:
        pass

    liquid_df = pd.DataFrame(index=component_list)
    liquid_df["mol"] = pd.Series(liquid)
    liquid_df['mol pct'] = liquid_df['mol'] / liquid_df['mol'].sum() *100
    liquid_pct_df = liquid_df['mol pct']
    liquid_pct_df = pd.DataFrame(liquid_pct_df)

    # handeling components with a zero conc
    vapor = eos.y
    if component_list != list(vapor.keys()):
        missing_comp_v = [x for x in component_list if x not in vapor]
        zeros = []
        for i in range(len(missing_comp_v)):
            i = 0
            zeros.append(i)
        temp_v = dict(zip(missing_comp_v, zeros))
        vapor = vapor.to_dict()
        vapor.update(temp_v)
    else:
        pass

    vapor_df = pd.DataFrame(index=component_list)
    vapor_df["mol"] = pd.Series(vapor)
    vapor_df['mol pct'] = vapor_df['mol'] / vapor_df['mol'].sum() *100
    vapor_pct_df = vapor_df['mol pct']
    vapor_pct_df = pd.DataFrame(vapor_pct_df)


    N2 =liquid_pct_df.iloc[:,0][0]
    CO2 = liquid_pct_df.iloc[:,0][2]
    C1 = liquid_pct_df.iloc[:,0][1]
    C2 = liquid_pct_df.iloc[:,0][3]
    C3 = liquid_pct_df.iloc[:,0][4]
    iC4 =  liquid_pct_df.iloc[:,0][5]
    C4 =   liquid_pct_df.iloc[:,0][6]
    iC5 = liquid_pct_df.iloc[:,0][7]
    C5 = liquid_pct_df.iloc[:,0][8]
    C6 = liquid_pct_df.iloc[:,0][9]
    C7 = liquid_pct_df.iloc[:,0][10]

    return N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, liquid_pct_df, vapor_pct_df



#@EOS_app.route('/')
#def home():
#    return render_template('index_EOS.html')



@EOS_app.route('/',methods=['GET', 'POST'])
def home():

    '''
    For rendering results on HTML GUI
    '''
    n_stages = 0
    shrink = 1


    ## added this so that everything could be added in the same root path
    if request.method == 'POST':
        ## data from form
        n_stages = int(request.form['stage'])


    #    Sample_name = request.form['id']
        SG = float(request.form['SG'])
        C7_SG = float(request.form['C7_plus_SG'])
        C7_MW = float(request.form['C7_plus_MW'])
        C1 = float(request.form['C1'])
        C2 = float(request.form['C2'])
        C3 = float(request.form['C3'])
        iC4 = float(request.form['iC4'])
        C4 =float(request.form['C4'])
        iC5 = float(request.form['iC5'])
        C5 = float(request.form['C5'])
        C6 = float(request.form['C6'])
        C7 = float(request.form['C7'])
        CO2 = float(request.form['CO2'])
        N2 = float(request.form['N2'])
        units = request.form['units']  #will equal: LV or Mol
    #    timestamp = ' {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    #    sample_time = Sample_name + timestamp

        P = float(request.form['P'])
        T = float(request.form['T'])

        if units == "LV":
            data = [N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, 1, P, T, C7_SG, SG, C7_MW]
            names = ['LV % N2', 'LV % CO2', 'LV % C1', 'LV % C2',
                'LV % C3', 'LV % IC4', 'LV % NC4', 'LV % IC5', 
                'LV % NC5', 'LV % C6', 'LV % C7', 'Shrinkage Factor',
                'Pressure', 'Temp', 'C7+ Specific Gravity', 'Total Spec. Grav.', 
                'C7+ Molecular Weight']
            df = pd.DataFrame([data], columns=names)

            sample1 = LV_Input(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, 1, P, T, C7_SG, SG, C7_MW)
            mole_fraction = sample1.mol_pct()
            mole_fraction = mole_fraction[0:11]
            component_list = [
                        'Nitrogen',
                        'Methane',
                        'Carbon Dioxide',
                        'Ethane',
                        'Propane',
                        'Isobutane',
                        'n-Butane',
                        'Isopentane',
                        'n-Pentane',
                        'n-Hexane',
                        'Heptanes+'
                        ]

            MW_plus_fraction = float(C7_MW)
            #MW_plus_fraction = 181
            SG_plus_fraction = float(C7_SG)
            #SG_plus_fraction = 0.795


            T = (T - 32) * 5 / 9 + 273.15
            P = (P + 14.65)/14.5038

            try:
                eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
                shrinkage = eos.shrinkage(bubble_point_force=False)
            except:
                shrinkage = "error"

            df['results VTPR'] = shrinkage


            ## ----------------      Preping Data for NN    --------------------------
            Light_Frac = df['LV % CO2'] + df['LV % C1'] + df['LV % C2'] +  df['LV % C3']
            Light_Frac = Light_Frac.values
            df['Light_Frac'] = Light_Frac

            #Heavy Fraction - 
            Heavy_Frac = df['LV % IC5'] + df['LV % NC5'] + df['LV % C6'] + df['LV % C7']
            Heavy_Frac = Heavy_Frac.values
            df['Heavy_Frac'] = Heavy_Frac

            #getting API gravity
            API_grav = df['Total Spec. Grav.']
            API_grav = API_grav.values
            API_grav = 141.5/API_grav - 131.5
            df['API_grav'] = API_grav

            Data = df[['Shrinkage Factor','Pressure', 'LV % IC4',
                                    'LV % NC4','C7+ Specific Gravity','Light_Frac','Heavy_Frac','API_grav']]
            X = Data.drop('Shrinkage Factor', axis = 1)

            Shrink_Model = keras.models.load_model(r'C:\Users\Preston\Shrinkage Model\shrinkage_nn_conoco_data_52620.h5')
            deep_prediction = Shrink_Model.predict(X).flatten()
            if deep_prediction > 1:
                deep_prediction = 1

            Wide_Model = keras.models.load_model(r'C:\Users\Preston\Shrinkage Model\VTPR NN\wide_NN_testing_61120_otherVTPRmodel.h5')
            wide_prediction = Wide_Model.predict((df['results VTPR'].values, X)).flatten()
            if wide_prediction > 1:
                wide_prediction = 1 



                ## ----------------        Constructing Shrinkage DF      --------------------------
            wide = float(wide_prediction)
            deep = float(deep_prediction)
            shrink_comb = {'Shrinkage': [shrinkage, deep, wide]}               
            shrink_df = pd.DataFrame(shrink_comb, index=['VTPR-EOS', '3 Layer NN', 'Wide NN'])
            shrink_dict = shrink_df.to_dict()['Shrinkage']
            for key, value in shrink_dict.items():
                shrink_dict[key] = round(value, 4)


        ## ---------------------------- Conditional for Mol input -----------------------------------------
            
        if units == "Mol":
            sample1 = Mol_Input(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, 1, P, T, C7_SG, SG, C7_MW)

            mole_fraction = sample1.mol_pct()
            mole_fraction = mole_fraction[0:11]

            data = sample1.lv_pct()
            names = ['LV % N2', 'LV % CO2', 'LV % C1', 'LV % C2',
                'LV % C3', 'LV % IC4', 'LV % NC4', 'LV % IC5', 
                'LV % NC5', 'LV % C6', 'LV % C7', 'Shrinkage Factor',
                'Pressure', 'Temp', 'C7+ Specific Gravity', 'Total Spec. Grav.', 
                'C7+ Molecular Weight']
            df = pd.DataFrame([data], columns=names)

            component_list = [
                        'Nitrogen',
                        'Methane',
                        'Carbon Dioxide',
                        'Ethane',
                        'Propane',
                        'Isobutane',
                        'n-Butane',
                        'Isopentane',
                        'n-Pentane',
                        'n-Hexane',
                        'Heptanes+'
                        ]

            MW_plus_fraction = float(C7_MW)
            #MW_plus_fraction = 181
            SG_plus_fraction = float(C7_SG)
            #SG_plus_fraction = 0.795


            T = (T - 32) * 5 / 9 + 273.15
            P = (P + 14.65)/14.5038

            try:
                eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
                shrinkage = eos.shrinkage(bubble_point_force=False)
            except:
                shrinkage = "error"
            df['results VTPR'] = shrinkage


            ## ----------------      Preping Data for NN    --------------------------


            Light_Frac = df['LV % CO2'] + df['LV % C1'] + df['LV % C2'] +  df['LV % C3']
            Light_Frac = Light_Frac.values
            df['Light_Frac'] = Light_Frac

            #Heavy Fraction - 
            Heavy_Frac = df['LV % IC5'] + df['LV % NC5'] + df['LV % C6'] + df['LV % C7']
            Heavy_Frac = Heavy_Frac.values
            df['Heavy_Frac'] = Heavy_Frac

            #getting API gravity
            API_grav = df['Total Spec. Grav.']
            API_grav = API_grav.values
            API_grav = 141.5/API_grav - 131.5
            df['API_grav'] = API_grav

            Data = df[['Shrinkage Factor','Pressure', 'LV % IC4',
                                    'LV % NC4','C7+ Specific Gravity','Light_Frac','Heavy_Frac','API_grav']]
            X = Data.drop('Shrinkage Factor', axis = 1)


            Shrink_Model = keras.models.load_model(r'C:\Users\Preston\Shrinkage Model\shrinkage_nn_conoco_data_52620.h5')
            deep_prediction = Shrink_Model.predict(X).flatten()
            if deep_prediction > 1:
                deep_prediction = 1
                
            Wide_Model = keras.models.load_model(r'C:\Users\Preston\Shrinkage Model\VTPR NN\wide_NN_testing_61120_otherVTPRmodel.h5')
            wide_prediction = Wide_Model.predict((df['results VTPR'].values, X)).flatten()
            if wide_prediction > 1:
                wide_prediction = 1 


            ## ----------------        Constructing Shrinkage DF      --------------------------
            wide = float(wide_prediction)
            deep = float(deep_prediction)
            shrink_comb = {'Shrinkage': [shrinkage, deep, wide]}               
            shrink_df = pd.DataFrame(shrink_comb, index=['VTPR-EOS', '3 Layer NN', 'Wide NN'])
            shrink_dict = shrink_df.to_dict()['Shrinkage']
            for key, value in shrink_dict.items():
                shrink_dict[key] = round(value, 4)





        print("ran this many stages", n_stages)
        ## -------------------  Multistage flash calcs ---------------------------------------

        if n_stages == 1:
            single_stage = final_stage(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW)
            liquid_out = single_stage[11]
            liquid_out_dict = liquid_out.to_dict()['mol pct'] 
            for key, value in liquid_out_dict.items():
                liquid_out_dict[key] = round(value, 4)

            vapor_out = single_stage[12]
            vapor_out_dict = vapor_out.to_dict()['mol pct']
            for key, value in vapor_out_dict.items():
                vapor_out_dict[key] = round(value, 4)

            n1_liquid = liquid_out.to_html(header = 'true')
            n1_vapor = vapor_out.to_html(header = 'true')
            print("ran stage1")
        
            return render_template('section.html',
            n1_liquid = liquid_out_dict, n1_vapor = vapor_out_dict, summary_dict = shrink_dict)
            #summary = shrink_df.to_html(header = 'true'))    
        
        elif n_stages == 2:
            P2 = float(request.form['P2'])
            T2 = float(request.form['T2'])
            stage1 = n_stage(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW, P2, T2)
            
            liquid_out_stage1 = stage1[11]
            liquid_out_dict1 = liquid_out_stage1.to_dict()['mol pct']
            for key, value in liquid_out_dict1.items():
                liquid_out_dict1[key] = round(value, 4)

            vapor_out_stage1 = stage1[12]
            vapor_out_dict1 = vapor_out_stage1.to_dict()['mol pct']
            for key, value in vapor_out_dict1.items():
                vapor_out_dict1[key] = round(value, 4)           

            stage2 = final_stage(stage1[0], stage1[1], stage1[2], stage1[3], stage1[4], stage1[5], 
                    stage1[6], stage1[7],  stage1[8],  stage1[9],  stage1[10], shrink, P2, T2, SG, C7_SG, C7_MW)
            
            liquid_out_stage2 = stage2[11]
            liquid_out_dict2 = liquid_out_stage2.to_dict()['mol pct']
            for key, value in liquid_out_dict2.items():
                liquid_out_dict2[key] = round(value, 4)

            vapor_out_stage2 = stage2[12]
            vapor_out_dict2 = vapor_out_stage2.to_dict()['mol pct']
            for key, value in vapor_out_dict2.items():
                vapor_out_dict2[key] = round(value, 4) 
            
            return render_template('section.html',
            n1_liquid = liquid_out_dict1, 
            n1_vapor = vapor_out_dict1,
            n2_liquid = liquid_out_dict2,
            n2_vapor = vapor_out_dict2,
            summary_dict = shrink_dict)
        
        
        
        
        elif n_stages == 3:
            P2 = float(request.form['P2'])
            T2 = float(request.form['T2'])
            T3 = float(request.form['T3'])
            P3 = float(request.form['P3'])
            stage1 = n_stage(N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, shrink, P, T, C7_SG, SG, C7_MW, P2, T2)
            
            liquid_out_stage1 = stage1[11]
            liquid_out_dict1 = liquid_out_stage1.to_dict()['mol pct']
            for key, value in liquid_out_dict1.items():
                liquid_out_dict1[key] = round(value, 4)

            vapor_out_stage1 = stage1[12]
            vapor_out_dict1 = vapor_out_stage1.to_dict()['mol pct']
            for key, value in vapor_out_dict1.items():
                vapor_out_dict1[key] = round(value, 4)           

            stage2 = n_stage(stage1[0], stage1[1], stage1[2], stage1[3], stage1[4], stage1[5], 
                    stage1[6], stage1[7],  stage1[8],  stage1[9],  stage1[10], shrink, P2, T2, SG, C7_SG, C7_MW, P3, T3)
            
            liquid_out_stage2 = stage2[11]
            liquid_out_dict2 = liquid_out_stage2.to_dict()['mol pct']
            for key, value in liquid_out_dict2.items():
                liquid_out_dict2[key] = round(value, 4)

            vapor_out_stage2 = stage2[12]
            vapor_out_dict2 = vapor_out_stage2.to_dict()['mol pct']
            for key, value in vapor_out_dict2.items():
                vapor_out_dict2[key] = round(value, 4) 
            
            stage3 = final_stage(stage2[0], stage2[1], stage2[2], stage2[3], stage2[4], stage2[5],
                                    stage2[6], stage2[7],  stage2[8],  stage2[9],  stage2[10], shrink, P3, T3, C7_SG, SG, C7_MW)
            
            liquid_out_stage3 = stage3[11] 
            liquid_out_dict3 = liquid_out_stage3.to_dict()['mol pct']
            for key, value in liquid_out_dict3.items():
                liquid_out_dict3[key] = round(value, 4)

            vapor_out_stage3 = stage3[12]
            vapor_out_dict3 = vapor_out_stage3.to_dict()['mol pct']
            for key, value in vapor_out_dict3.items():
                vapor_out_dict3[key] = round(value, 4) 

            return render_template('section.html',
            n1_liquid = liquid_out_dict1, 
            n1_vapor = vapor_out_dict1,
            n2_liquid = liquid_out_dict2,
            n2_vapor = vapor_out_dict2,
            n3_liquid = liquid_out_dict3,
            n3_vapor = vapor_out_dict3,
            summary_dict = shrink_dict)
        
        else:
            return jsonfiy({'error' : 'An Error Occured'})




    ## returning default html page
    else:
        return render_template('index_multi.html')
    #    table1 = liquid_pct_df.to_html(header = 'true'), 
    #    table2 = vapor_pct_df.to_html(header = 'true'),
    #    table3 = shrink_df.to_html(header = 'true'))
            
