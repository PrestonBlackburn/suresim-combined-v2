  
from flask import request, jsonify, render_template, Blueprint
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math 
import datetime
import numpy as np
import joblib
import pandas as pd

from VTPR import VTPR
from pandas import read_csv, DataFrame
import time
##import os

##os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


shrinkage_csv = Blueprint("shrinkage_csv", __name__, static_folder = "static", template_folder = "templates")
Shrink_Model = keras.models.load_model('shrinkage_nn_conoco_data_52620.h5')
Wide_Model = keras.models.load_model('wide_NN_testing_61120_otherVTPRmodel.h5')

@shrinkage_csv.route('/')
def home():
    return render_template('index_shrink.html')

@shrinkage_csv.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    file = request.files['inputFile']
    Data = pd.read_csv(file)
    
    ## ----        Adding VTPR Calculations      ------

    df = Data
    names =  ['Nitrogen', 'Carbon Dioxide', 'Methane',  
          'Ethane', 'Propane', 'Isobutane', 'n-butane', 
          'Isopentane', 'n-pentane', 'hexanes', 'heptanes+']

    Density = [6.727, 6.8129, 2.5, 2.9704 , 4.2285, 4.6925, 4.8706, 5.212, 5.2584, 5.5364]
    MW = [28.0, 44.0, 16.0, 30.1, 44.1, 58.1, 58.1, 72.1, 72.1, 86.2]

    LV_input = df[['LV % N2', 'LV % CO2', 'LV % C1', 'LV % C2',
                'LV % C3', 'LV % IC4', 'LV % NC4', 'LV % IC5', 
                'LV % NC5', 'LV % C6', 'LV % C7']]
    LV_sum = np.sum(LV_input, axis = 1)
    norm_LV = (LV_input.values/LV_sum[:,None])*100
    norm_LV = pd.DataFrame(norm_LV, columns = names)

    Density_df = norm_LV.iloc[:,0:10] * Density
    Density_df['heptanes+'] = norm_LV['heptanes+'] * df['C7+ Specific Gravity']*8.3372
    Density_sum = np.sum(Density_df, axis=1)
    W_pct = (Density_df/Density_sum[:,None])*100

    MW_df = W_pct.iloc[:,0:10] / MW
    MW_df['heptanes+'] = W_pct['heptanes+'] / df['C7+ Molecular Weight']
    MW_sum = np.sum(MW_df, axis=1)
    mol_pct = (MW_df/MW_sum[:,None])*100

    results = []

    for i in range(0, len(df)):
        try:
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

            mole_fraction = mol_pct.iloc[i].values

            T_df = df['Temp']
            T = T_df[i]

            #T = 145
            P_df = df['Pressure']
            P = P_df[i]

            #P = 1176
            MW_df = df['C7+ Molecular Weight']
            SG_df = df['C7+ Specific Gravity']

            MW_plus_fraction = float(MW_df[i])
            #MW_plus_fraction = 181
            SG_plus_fraction = float(SG_df[i])
            #SG_plus_fraction = 0.795


            T = (T - 32) * 5 / 9 + 273.15
            P = (P + 14.65)/14.5038
            eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
            shrinkage = eos.shrinkage(bubble_point_force=False)
            results.append(shrinkage)
        except:
            shrinkage = "error"
            results.append(shrinkage)

        VTPR_Results = pd.DataFrame(results, columns = ['results'])
    ##   ------   End of calculations for VTPR   ---------


    ## --------- Preparing the data for the Neural Net ---------

    Well_name = Data['Well Name']

    Light_Frac = Data['LV % CO2'] + Data['LV % C1'] + Data['LV % C2'] +  Data['LV % C3']
    Light_Frac = Light_Frac.values
    Data['Light_Frac'] = Light_Frac

    #Heavy Fraction - 
    Heavy_Frac = Data['LV % IC5'] + Data['LV % NC5'] + Data['LV % C6'] + Data['LV % C7']
    Heavy_Frac = Heavy_Frac.values
    Data['Heavy_Frac'] = Heavy_Frac

    #getting API gravity
    API_grav = Data['Total Spec. Grav.']
    API_grav = API_grav.values
    API_grav = 141.5/API_grav - 131.5
    Data['API_grav'] = API_grav

    Data = Data[['Shrinkage Factor','Pressure', 'LV % IC4',
                            'LV % NC4','C7+ Specific Gravity','Light_Frac','Heavy_Frac','API_grav']]

    X = Data.drop('Shrinkage Factor', axis=1)
    y = Data['Shrinkage Factor']


    ##       ---- Modeling with Keras/tf  -------
    Prediction = Shrink_Model.predict(X).flatten()

    ##      -----  loop to catch some VTPR EOS errors    -------
    error_check = sum(VTPR_Results['results'].astype(str).str.count('error').values)
    print(error_check)

    if error_check > 0:
        test_df = pd.DataFrame(Prediction, columns = ['Neural Net'])
        test_df['VTPR EOS'] = VTPR_Results['results']
        test_df['Real Values'] = y
        test_df['Well name'] = Well_name

        MAE_NN = mean_absolute_error(y, Prediction)
        MSE_NN = mean_squared_error(y, Prediction)
        RMSE_NN = MSE_NN ** (1/2)

        stat_dict = {
            'Model':['Neural Net'],
            'MAE': [MAE_NN],
            'RMSE': [RMSE_NN]
        }

        EOS_stats = pd.DataFrame(stat_dict)

    else: 
        Wide_Prediction = Wide_Model.predict((VTPR_Results , X)).flatten()
        test_df = pd.DataFrame(Prediction, columns = ['Neural Net'])
        test_df['Wide NN'] = Wide_Prediction
        test_df['VTPR EOS'] = VTPR_Results['results']
        test_df['Real Values'] = y
        test_df['Well name'] = Well_name

        MAE_NN = mean_absolute_error(y, Prediction)
        MSE_NN = mean_squared_error(y, Prediction)
        RMSE_NN = MSE_NN ** (1/2)

        MAE_wide = mean_absolute_error(y, Wide_Prediction)
        MSE_wide = mean_squared_error(y, Wide_Prediction)
        RMSE_wide = MSE_wide ** (1/2)

        MAE_VTPR = mean_absolute_error(y, results)
        MSE_VTPR = mean_squared_error(y, results)
        RMSE_VTPR = MSE_VTPR ** (1/2)

        stat_dict = {
        'Model': ["Neural Net", "Wide Neural Net", "VTPR"],
        'MAE' :[MAE_NN, MAE_wide, MAE_VTPR],
        'RMSE': [RMSE_NN, RMSE_wide, RMSE_VTPR]
        }

        EOS_stats = pd.DataFrame(stat_dict)
        


    test_df.head()
    test_df = test_df.dropna()


    ##   stats calcs





    #mae = mean_absolute_error(test_df['Prediction'].values, test_df['Real Values'].values)
    #mse = mean_squared_error(test_df['Prediction'].values, test_df['Real Values'].values)
    #rmse = mse ** 0.5

    return render_template('index_shrink.html', table1 = test_df.to_html(header = 'true'),
                            table2 = EOS_stats.to_html(header = 'true'))

