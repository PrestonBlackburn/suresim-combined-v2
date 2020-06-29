import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import math 
from VTPR import VTPR
import datetime
import os
from flask import make_response, request, jsonify, render_template, Blueprint


from flask import Flask, request


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
        norm_mol_vals = norm_mol.values
        
        self.N2 = norm_mol_vals[0,0]
        self.CO2 = norm_mol_vals[0,1]
        self.C1 = norm_mol_vals[0,2]
        self.C2 = norm_mol_vals[0,3]
        self.C3 = norm_mol_vals[0,4]
        self.iC4 = norm_mol_vals[0,5]
        self.C4 = norm_mol_vals[0,6]
        self.iC5 = norm_mol_vals[0,7]
        self.C5 = norm_mol_vals[0,8]
        self.C6 = norm_mol_vals[0,9]
        self.C7 = norm_mol_vals[0,10]        
        
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

        Density_df = W_pct.iloc[:,0:10] / Mol_Input.Density
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


EOS_api_app = Blueprint("EOS_api_app", __name__, static_folder = "static", template_folder = "templates")



@EOS_api_app.route('/', methods = ['POST', 'GET'])
def json_example():
    req_data = request.get_json()
    JSON_data = req_data


    sample2 = Mol_Input.from_dict(JSON_data)

    mole_fraction = sample2.mol_pct()

    C7_SG = mole_fraction[14]
    C7_MW = mole_fraction[16]
    T = mole_fraction[13]
    P = mole_fraction[12]
    print(T)
    print(P)


    mole_fraction = mole_fraction[0:11]
    print(mole_fraction)

    data = sample2.lv_pct()
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

    #try:
    print(component_list)
    print(mole_fraction)
    eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
    shrinkage = eos.shrinkage(bubble_point_force=False)
    #except:
    #    shrinkage = "error"

    print(shrinkage)
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


    ##   -------------------- Handling Liquid and Vapor Streams   -------------------------
    Density = [6.727, 6.8129, 2.5, 2.9704 , 4.2285, 4.6925, 4.8706, 5.212, 5.2584, 5.5364, float(df['C7+ Specific Gravity'].values*8.3372)]
    MW = [28.0, 44.0, 16.0, 30.1, 44.1, 58.1, 58.1, 72.1, 72.1, 86.2, float(df['C7+ Molecular Weight'].values)]

    density = np.reshape(Density, (1, 11)).T
    mw = np.reshape(MW, (1,11)).T

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


    shrink_df_dict = shrink_df.to_dict()
    liquid_out_dict = liquid_pct_df.to_dict()
    vapor_out_dict = vapor_pct_df.to_dict()
    combined_data = (shrink_df_dict, liquid_out_dict, vapor_out_dict)

    return make_response(jsonify(combined_data), 200)
         
