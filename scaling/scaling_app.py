from flask import Blueprint, request, jsonify, render_template
import numpy as np
import pandas as pd
import math 
import Scale
from Scale import Scaling_Calcs
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
 
       

        scale_results = Scaling_Calcs(Na, K, Ca, Mg, Ba, Sr, Fe, Mn, Cl, F, HCO3, CO3, SO4, NO3, Temp_C, pH, Silica)

        K_sp_vals = Scaling_Calcs.Minerals['log(IP/K_sp)'].values
        Expected_precip = Scaling_Calcs.Minerals['Expected_Precipitate_per_1000_bbls'].values
        Index_mins = ["Calcite", "Aragonite", "FeCO3", "MgCO3", "BaSO4", "CaSO4", "SrSO4", "CaF2"]
        data_mins = {'log(IP/K_sp)' : K_sp_vals, 'Precipitate per 1000 bbls' : Expected_precip, 'Compound Names' : Index_mins}
        df_mins = pd.DataFrame.from_dict(data_mins)
        df_mins = df_mins.set_index("Compound Names")

        Precipitate = Scaling_Calcs.Minerals['Expected_Precipitate_per_1000_bbls'].reset_index()
        Scale_Indicies = Scaling_Calcs.Scale_Indicies


        Minerals_Temp_Range = []
        Temp = range(15, 100, 5)

        for Temp_C in Temp:
            scale_results = Scaling_Calcs(Na, K, Ca, Mg, Ba, Sr, Fe, Mn, Cl, F, HCO3, CO3, SO4, NO3, Temp_C, pH, Silica)
            mineral_array = Scaling_Calcs.Minerals['log(IP/K_sp)'].values
            Minerals_Temp_Range.append((mineral_array))

        Temp = list(range(15, 100, 5))
        Temp_minerals = Minerals_Temp_Range
        Temp_minerals_df = pd.DataFrame((Temp_minerals))
        #Temp_minerals_df = Temp_minerals_df.transpose()
        Temp_minerals_df.columns = ["Calcite", "Aragonite", "FeCO3", "MgCO3", "BaSO4", "CaSO4", "SrSO4", "CaF2"]
        Temp_minerals_df.index = list(Temp)
        
        Calcite_ksp = Temp_minerals_df['Calcite'].values
        Calcite_ksp  = Calcite_ksp.tolist()
        labels = list(Temp)
        Aragonite_ksp = Temp_minerals_df['Aragonite'].values
        Aragonite_ksp  = Aragonite_ksp.tolist()

        Ions_dict = {"Na_+" : Na, "K_+" : K, "Ca_2+" : Ca, "Mg_2+" : Mg, "Ba_2+" : Ba, "Sr_2+" : Sr,
                "Fe_2+" : Fe, "Mn_2+" : Mn, "Cl_-" : Cl, "F_-" : F, "HCO3_-" : HCO3, "CO3_2-" : CO3,
            "SO4_2-" : SO4, "NO3_-" : NO3}
        df_ions = pd.DataFrame(Ions_dict, index = [0])
        df_ions = df_ions.transpose()


    #### For uploaded data + overwriting input values:
    if Sample_name == None:
    #    file = request.files['inputFile']
    #    data = pd.read_csv(file)
    #    vals = data.drop(['Sample Name','Temperature','Pressure'], axis=1)
        pass



    


    return render_template('Scale_index.html', table1 = Scale_Indicies.to_html(header = 'true'),
     table2 = df_mins.to_html(header = 'true'), table3 = Temp_minerals_df.to_html(header = 'true'),
     table4 = df_ions.to_html(header = 'true'),  Calcite_ksp =Calcite_ksp,
      Aragonite_ksp = Aragonite_ksp, labels=labels)
   

