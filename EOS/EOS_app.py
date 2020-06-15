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


'''#####  Database info
#for switching between horoku and local
ENV = 'prod'
if ENV == 'dev':
    app.debug=True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://postgres:1357@localhost/thermo'
else:
    app.debug=False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://jsspeqhnqwnpbr:85a4fa4e2f414a0547beaf8fb20138b1a72e53d30ae6cb249f85c40f5482bf9d@ec2-50-17-178-87.compute-1.amazonaws.com:5432/d227h6cf7mg0a2'

#add this so you don't get a warning in the consol
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#create database object
db = SQLAlchemy(app)

class Parameters(db.Model):
    __tablename__ = 'parameters'
    Sample_name = db.Column(db.String(400), primary_key = True)
    Spec_grav = db.Column(db.Float)
    Shrinkage = db.Column(db.Float)
    Nitrogen_LV = db.Column(db.Float)
    Carbon_Dioxide_LV = db.Column(db.Float)
    Methane_LV = db.Column(db.Float)
    Ethane_LV = db.Column(db.Float)
    Propane_LV = db.Column(db.Float)
    Isobutane_LV = db.Column(db.Float)
    nButane_LV = db.Column(db.Float)
    Isopentane_LV = db.Column(db.Float)
    nPentane_LV = db.Column(db.Float)
    Hexanes_LV = db.Column(db.Float)
    Heptanes_Plus_LV = db.Column(db.Float)
    Input_Pressure = db.Column(db.Float)
    Input_Temperature = db.Column(db.Float)

    def __init__(self, Sample_name, Shrinkage, Spec_grav, Nitrogen_LV, Carbon_Dioxide_LV, 
    Methane_LV, Ethane_LV, Propane_LV, Isobutane_LV, nButane_LV, Isopentane_LV, nPentane_LV, Hexanes_LV,
    Heptanes_Plus_LV, Input_Pressure, Input_Temperature):
       self.Sample_name = Sample_name
       self.Shrinkage = Shrinkage
       self.Spec_grav = Spec_grav
       self.Nitrogen_LV = Nitrogen_LV
       self.Carbon_Dioxide_LV = Carbon_Dioxide_LV
       self.Methane_LV = Methane_LV
       self.Ethane_LV = Ethane_LV
       self.Propane_LV = Propane_LV
       self.Isobutane_LV = Isobutane_LV
       self.nButane_LV = nButane_LV
       self.Isopentane_LV = Isopentane_LV
       self.nPentane_LV = nPentane_LV
       self.Hexanes_LV = Hexanes_LV
       self.Heptanes_Plus_LV = Heptanes_Plus_LV
       self.Input_Pressure = Input_Pressure
       self.Input_Temperature = Input_Temperature




#### End most of database stuff

#### calling + using the EOS model'''



@EOS_app.route('/')
def home():
    return render_template('index_EOS.html')



@EOS_app.route('/predict',methods=['POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''

    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = sum(final_features)
    
    #html_input = [float(x) for x in request.form.values()]

    #output = round(prediction[0], 4)
    Sample_name = request.form['id']
    SG = float(request.form['SG'])
    P = float(request.form['P'])
    T = float(request.form['T'])
    C7_SG = float(request.form['C7+ SG'])
    C7_MW = float(request.form['C7+ MW'])
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

    T_sep = 60
    P_sep = 0
    T_feed = T
    P_feed = P
    timestamp = ' {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    sample_time = Sample_name + timestamp



    data = [N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, 1, P, T, C7_SG, SG, C7_MW]

    names = ['LV % N2', 'LV % CO2', 'LV % C1', 'LV % C2',
        'LV % C3', 'LV % IC4', 'LV % NC4', 'LV % IC5', 
        'LV % NC5', 'LV % C6', 'LV % C7', 'Shrinkage Factor',
        'Pressure', 'Temp', 'C7+ Specific Gravity', 'Total Spec. Grav.', 
        'C7+ Molecular Weight']


    ##   -------------      VTPR Calcs + Unit Conversions     --------------------

    df = pd.DataFrame([data], columns=names)

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
            print(shrinkage)
            results.append(shrinkage)
        except:
            shrinkage = "error"
            print(shrinkage)
            results.append(shrinkage)   

    df['results VTPR'] = shrinkage


    ## ----------------      NN Calcs + Data Prep   --------------------------


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

    Shrink_Model = keras.models.load_model('shrinkage_nn_conoco_data_52620.h5')
    deep_prediction = Shrink_Model.predict(X).flatten()
    if deep_prediction > 1:
        deep_prediction = 1

    Wide_Model = keras.models.load_model('wide_NN_testing_61120_otherVTPRmodel.h5')
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

    liquid = eos.x
    liquid_df = pd.DataFrame.from_dict(liquid)
    liquid_df.columns = ['mol']
    liquid_df['density'] = density
    liquid_df['MW'] = mw
    liquid_df['Wt'] = liquid_df['mol'] * liquid_df['MW']
    liquid_df['Wt_pct'] = liquid_df['Wt'] / liquid_df['Wt'].sum()
    liquid_df['LV'] = liquid_df['Wt_pct'] / liquid_df['density']
    liquid_df['LV_pct'] = liquid_df['LV'] / liquid_df['LV'].sum() *100
    liquid_pct_df = liquid_df['LV_pct']
    liquid_pct_df = pd.DataFrame(liquid_pct_df)

    vapor = eos.y
    vapor_df = pd.DataFrame.from_dict(vapor)
    vapor_df.columns = ['mol']
    vapor_df['density'] = density
    vapor_df['MW'] = mw
    vapor_df['Wt'] = vapor_df['mol'] * vapor_df['MW']
    vapor_df['Wt_pct'] = vapor_df['Wt'] / vapor_df['Wt'].sum()
    vapor_df['LV'] = vapor_df['Wt_pct'] / vapor_df['density']
    vapor_df['LV_pct'] = vapor_df['LV'] / vapor_df['LV'].sum() *100
    vapor_pct_df = vapor_df['LV_pct']
    vapor_pct_df = pd.DataFrame(vapor_pct_df)



    '''
    #more database code
    database_data = Parameters(sample_time, output, SG, N2, CO2, C1, C2, C3, iC4, C4, iC5, C5, C6, C7, P_feed, T_feed)
    db.session.add(database_data)
    db.session.commit()
    #end database code
    '''

    return render_template('index_EOS.html',
    table1 = liquid_pct_df.to_html(header = 'true'), 
    table2 = vapor_pct_df.to_html(header = 'true'),
    table3 = shrink_df.to_html(header = 'true'))
         
