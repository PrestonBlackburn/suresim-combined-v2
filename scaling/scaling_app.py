from flask import Blueprint, request, jsonify, render_template
import numpy as np
import pandas as pd
import math 
import Scale
from Scale import  Scale_Calcs_dT
import datetime
import json

scaling_app = Blueprint("scaling_app", __name__, static_folder = "static", template_folder = "templates")



@scaling_app.route('/')
def home():
    return render_template('Scale_index.html')

@scaling_app.route('/predict',methods=['POST'])
def predict():

    Sample_name = request.form.get('id')
    
    if Sample_name != None:

        Na = request.form['Na']
        Na = float(Na)
        K = request.form['K']
        K = float(K)
        Ca = request.form['Ca']
        Ca = float(Ca)
        Mg = request.form['Mg']
        Mg = float(Mg)
        Ba = request.form['Ba']
        Ba = float(Ba)
        Sr = request.form['Sr']
        Sr = float(Sr)
        Fe = request.form['Fe']
        Fe = float(Fe)
        Mn = request.form['Mn']
        Mn = float(Mn)
        Cl = request.form['Cl']
        Cl = float(Cl)
        F = request.form['F']
        F = float(F)
        HCO3 = request.form['HCO3']
        HCO3 = float(HCO3)
        CO3 = request.form['CO3']
        CO3 = float(CO3)
        SO4 = request.form['SO4']
        SO4 = float(SO4)
        NO3 = request.form['NO3']
        NO3 = float(NO3)
        Temp_C = request.form['Temp_C']
        Temp_C = float(Temp_C)
        pH = request.form['pH']
        pH = float(pH)
        Silica = request.form['Silica']
        Silica = float(Silica)
        H2S = request.form['H2S']
        H2S = float(H2S)
 
       

        scale_results = Scale_Calcs_dT(Na, K, Ca, Mg, Ba, Sr, Fe, Mn, Cl, F, HCO3, CO3, SO4, NO3, Temp_C, pH, Silica, H2S)

        temp_range = range(15, 60, 2)
        percipitate = []
        sat_index = []
        scale_index = []
        for i in temp_range:
            scale_results = Scale_Calcs_dT(Na, K, Ca, Mg, Ba, Sr, Fe, Mn, Cl, F, HCO3, CO3, SO4, NO3, i, pH, Silica, H2S)
            percipitate.append(scale_results[2])
            sat_index.append(scale_results[3])
            scale_index.append(scale_results[4])


        percipitate_df_T = pd.DataFrame(percipitate, columns=scale_results[1], index=temp_range)
        percipitate_df_T.index.name = "Temperature (C)"
        sat_index_df_T = pd.DataFrame(sat_index, columns = scale_results[1], index = temp_range)
        sat_index_df_T.index.name = "Temperature (C)"
        scale_index_df_T = pd.DataFrame(scale_index, columns = scale_results[5], index=temp_range)
        scale_index_df_T.index.name = "Temperature (C)"

        ion_df = scale_results[0][["mg/L", "Mol/L", "m/kg Water"]]
        sat_index_df = pd.DataFrame([scale_results[3]], columns = scale_results[1])
        sat_index_df = sat_index_df.T
        sat_index_df.rename(columns = {0: "Saturation Index"}, inplace=True)
        precipt_df = pd.DataFrame([scale_results[2]], columns = scale_results[1])
        precipt_df =precipt_df.T
        sat_index_df["Ptb"] = precipt_df.values

        SI_df = pd.DataFrame([scale_results[4]], columns =scale_results[5])
        SI_df = SI_df.T
        SI_df.rename(columns = {0:"Scaling Indicies"}, inplace=True)

        #labels for Chartist
        labels = list(temp_range)
        Calcite_ksp = sat_index_df_T['Calcite'].values
        Calcite_ksp = Calcite_ksp.tolist()
        Aragonite_ksp = sat_index_df_T['Aragonite'].values
        Aragonite_ksp = Aragonite_ksp.tolist()
        CaSO4_ksp = sat_index_df_T['CaSO4'].values
        CaSO4_ksp = CaSO4_ksp.tolist()
        Gypsum_ksp = sat_index_df_T['Gypsum'].values
        Gypsum_ksp = Gypsum_ksp.tolist()
        BaSO4_ksp = sat_index_df_T['BaSO4'].values
        BaSO4_ksp = BaSO4_ksp.tolist()
        SrSO4_ksp = sat_index_df_T['SrSO4'].values
        SrSO4_ksp = SrSO4_ksp.tolist()
        FeCO3_ksp = sat_index_df_T['FeCO3'].values
        FeCO3_ksp = FeCO3_ksp.tolist()
        FeS_ksp = sat_index_df_T['FeS'].values
        FeS_ksp = FeS_ksp.tolist()
        Halite_ksp = sat_index_df_T['Halite'].values
        Halite_ksp = Halite_ksp.tolist()
        MgCO3_ksp = sat_index_df_T['MgCO3'].values
        MgCO3_ksp = MgCO3_ksp.tolist()
        CaF2_ksp = sat_index_df_T['CaF2'].values
        CaF2_ksp = CaF2_ksp.tolist()
 



        


    #### For uploaded data + overwriting input values:
    if Sample_name == None:
    #    file = request.files['inputFile']
    #    data = pd.read_csv(file)
    #    vals = data.drop(['Sample Name','Temperature','Pressure'], axis=1)
        pass


    return render_template('Scale_index.html', table1 = SI_df.to_html(header = 'true'),
     table2 =  sat_index_df.to_html(header = 'true'), table3 = sat_index_df_T.to_html(header = 'true'),
     table4 = ion_df.to_html(header = 'true'),  Calcite_ksp =Calcite_ksp,
      Aragonite_ksp = Aragonite_ksp, CaSO4_ksp = CaSO4_ksp, Gypsum_ksp = Gypsum_ksp,
      BaSO4_ksp = BaSO4_ksp, SrSO4_ksp = SrSO4_ksp, FeCO3_ksp = FeCO3_ksp, FeS_ksp = FeS_ksp,
      Halite_ksp = Halite_ksp, MgCO3_ksp = MgCO3_ksp, CaF2_ksp = CaF2_ksp, labels=labels)
   

