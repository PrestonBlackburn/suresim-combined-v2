import pandas as pd
import numpy as np
import math



def Scaling_Calcs(Na, K, Ca, Mg, Ba, Sr, Fe, Mn, Cl, F, HCO3, CO3, SO4, NO3, Temp_C, pH, Silica):
    
    ####Constants
    R_const = 8.314
    Temp_K = Temp_C + 273.15
    Temp_F = (Temp_C * 9/5) + 32
    Multiplier = [1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1]
    Ion_Conc_Multiplier = [1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 1]
    Ion_Charge = [1, 1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -2, -2, -1]
    alpha = [0.45, 0.30, 0.60, 0.80, 0.50, 0.50, 0.60, 0.60, 0.30, 0.35, 0.45, 0.45, 0.40, 0.30]
    MW = [22.9898, 39.10, 40.08, 24.31, 137.33, 87.62, 55.85, 54.94, 35.45, 19, 61.02, 60.01, 96.06, 62]
    MW_Minerals = [101.094, 101.094, 116.861, 85.321, 233.386, 136.134, 183.676, 78.074]

    Mineral_names = ["Calcite", "Aragonite", "FeCO3", "MgCO3", "BaSO4", "CaSO4", "SrSO4", "CaF2"]
    K_sp_const = [3.36e-9, 6.0e-9, 3.13e-9, 6.82e-6, 2.58e-9, 4.93e-5, 3.44e-7, 3.45e-11]
    Delta_G = [47420, 46420, 54370, 29510, 57010, 23720, 36970, 334960]
    Delta_H = [-13050, -12840, -25690, -48200, 26280, -17990, -1970, 421170]
    Gamma_Minerals = [0.25*0.19, 0.25*0.19, 0.25*0.19, 0.31*0.19, 0.21*0.17, 0.25*0.17, 0.21*0.17, 0.25*0.63**2]
    
    #dict format
    Ions_dict = {"Na_+" : Na, "K_+" : K, "Ca_2+" : Ca, "Mg_2+" : Mg, "Ba_2+" : Ba, "Sr_2+" : Sr,
            "Fe_2+" : Fe, "Mn_2+" : Mn, "Cl_-" : Cl, "F_-" : F, "HCO3_-" : HCO3, "CO3_2-" : CO3,
           "SO4_2-" : SO4, "NO3_-" : NO3}

    
    #Create DataFrame
    Ions = pd.DataFrame.from_dict(Ions_dict, orient = 'index', columns = ["mg/L"])


    #Adding Columns to Data Frame
    Ions["Multiplier"] = Multiplier
    Ions["Ion Conc Multiplier"] = Ion_Conc_Multiplier
    Ions["Ion Charge"] = Ion_Charge
    Ions["MW"] = MW
    Ions["Alpha"] = alpha


    #Additional Calculated Values
    TDS_calced = sum(Ions["mg/L"].values) + Silica
    Ions["meq/L"] = Ions["Multiplier"] * Ions["mg/L"] / Ions["MW"]
    Ions["m/kg Water"] = Ions["mg/L"] * 1000 / (Ions["MW"] *(1000000 - TDS_calced))
    Ions["Mol/L"] = Ions["mg/L"] / (Ions["MW"] * 1000) * Ions["Ion Conc Multiplier"]
    Total_Mol_per_Liter = sum(Ions["Mol/L"]) * 0.5
    Ions["gamma"] = 10 ** (-1* (0.509*(Ions["Ion Charge"]**2) * (Total_Mol_per_Liter)**(1/2)) /
                           (1 + 3.29*(Ions["Alpha"]) * (Total_Mol_per_Liter)**(1/2)) )

    ##Mineral solubility
    IP = [Ions.at['Ca_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L'], 
          Ions.at['Ca_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L'], 
         Ions.at['Fe_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L'], 
         Ions.at['Mg_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L'],
         Ions.at['Ba_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L'],
         Ions.at['Ca_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L'],
         Ions.at['Sr_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L'],
         Ions.at['Ca_2+', 'Mol/L'] * Ions.at['F_-', 'Mol/L'] * Ions.at['F_-', 'meq/L'] / 2000000000]

    Minerals = pd.DataFrame({"Minerals" : (Mineral_names) , "K_sp_const" : K_sp_const, 
                             "Delta_G" : Delta_G, "Delta_H" : Delta_H, "Gamma_Minerals": Gamma_Minerals})
    Minerals = Minerals.set_index('Minerals')
    Minerals["Delta_G_Temp"]  = (Minerals["Delta_H"]*((1/Temp_K) - (1/298.15)) + Minerals["Delta_G"] / 298.15) *  Temp_K
    Minerals["K_sp_temp"] = 2.71828 ** (-1 * Minerals["Delta_G_Temp"] / (R_const * Temp_K))
    Minerals["K_sp_T"] = Minerals["K_sp_temp"] / Minerals["Gamma_Minerals"]
    Minerals["IP"] = IP
    Minerals["log(IP/K_sp)"] = np.log10(Minerals.IP / Minerals.K_sp_T)
    Minerals["Conc_in_solution"] = Minerals["K_sp_T"] ** 0.5


    #Cation/Anion Conc Left to Precipitate + Conditionals
    Calcite_Cation = Ions.at['Ca_2+', 'Mol/L'] - Minerals.at['Calcite', 'Conc_in_solution']
    FeCO3_Cation = Ions.at['Fe_2+', 'Mol/L'] -  Minerals.at['FeCO3', 'Conc_in_solution']
    MgCO3_Cation = Ions.at['Mg_2+', 'Mol/L'] - Minerals.at['MgCO3', 'Conc_in_solution']
    SrSO4_Cation = Ions.at['Sr_2+', 'Mol/L'] - Minerals.at['SrSO4', 'Conc_in_solution']
    CaSO4_Cation = Ions.at['Ca_2+', 'Mol/L'] - Minerals.at['CaSO4', 'Conc_in_solution']

    if Minerals.at['BaSO4', 'Conc_in_solution'] <= 0:
        BaSO4_Cation = 0
    else:
        BaSO4_Cation = Ions.at['Ba_2+', 'Mol/L'] - Minerals.at['BaSO4', 'Conc_in_solution']

    BaSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['BaSO4', 'Conc_in_solution']
    FeCO3_Anion = Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['FeCO3', 'Conc_in_solution']

    if BaSO4_Anion > BaSO4_Cation:
        SrSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['SrSO4', 'Conc_in_solution'] - BaSO4_Cation
    else:
        SrSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['SrSO4', 'Conc_in_solution'] - BaSO4_Anion


    if BaSO4_Anion < 0:
        if SrSO4_Anion < 0:
            CaSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['CaSO4', 'Conc_in_solution'] - 0 - 0
        else:
            CaSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['CaSO4', 'Conc_in_solution'] - 0 - SrSO4_Anion
    else:
        if SrSO4_Anion < 0:
            CaSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['CaSO4', 'Conc_in_solution'] - BaSO4_Anion - 0
        else:
            CaSO4_Anion = Ions.at['SO4_2-', 'Mol/L'] - Minerals.at['CaSO4', 'Conc_in_solution'] - BaSO4_Anion - SrSO4_Anion


    if FeCO3_Anion > FeCO3_Cation:   
        Calcite_Anion =  (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['Calcite', 'Conc_in_solution'] - FeCO3_Cation)*0.64
    else:
        Calcite_Anion =  (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['Calcite', 'Conc_in_solution'] - FeCO3_Anion)*0.64

    if Calcite_Cation > Calcite_Anion:
        Aragonite_Cation =  Ions.at['Ca_2+', 'Mol/L'] - Minerals.at['Aragonite', 'Conc_in_solution'] - Calcite_Anion
    else:
        Aragonite_Cation =  Ions.at['Ca_2+', 'Mol/L'] - Minerals.at['Aragonite', 'Conc_in_solution'] - Calcite_Cation


    if FeCO3_Anion > FeCO3_Cation:
        if Calcite_Anion > Calcite_Cation:
            Aragonite_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['Aragonite', 'Conc_in_solution'] -
                               FeCO3_Cation - Calcite_Cation) * .36
        else:
            Aragonite_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['Aragonite', 'Conc_in_solution'] -
                               FeCO3_Cation - Calcite_Anion) * .36
    else:
        if Calcite_Anion > Calcite_Cation:
            Aragonite_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['Aragonite', 'Conc_in_solution'] -
                               FeCO3_Anion - Calcite_Cation) * .36
        else:
            Aragonite_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['Aragonite', 'Conc_in_solution'] -
                               FeCO3_Anion - Calcite_Anion) * .36


    if Calcite_Cation > Calcite_Anion:
        if Aragonite_Cation > Aragonite_Anion:
            MgCO3_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['MgCO3', 'Conc_in_solution'] - FeCO3_Cation - 
            Calcite_Anion - Aragonite_Anion)
        else:
            MgCO3_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['MgCO3', 'Conc_in_solution'] - FeCO3_Cation -
            Calcite_Anion - Aragonite_Cation)
    else:
        if Aragonite_Cation > Aragonite_Anion:
            MgCO3_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['MgCO3', 'Conc_in_solution'] - FeCO3_Cation - 
            Calcite_Cation - Aragonite_Anion)
        else:
            MgCO3_Anion = (Ions.at['HCO3_-', 'Mol/L'] - Minerals.at['MgCO3', 'Conc_in_solution'] - FeCO3_Cation -
            Calcite_Cation - Aragonite_Cation)   

    CaF2_Cation = 0        
    CaF2_Anion = 0


    Cation_Array = [Calcite_Cation, Aragonite_Cation, FeCO3_Cation, MgCO3_Cation,
                    BaSO4_Cation, CaSO4_Cation, SrSO4_Cation, CaF2_Cation]
    Anion_Array = [Calcite_Anion, Aragonite_Anion, FeCO3_Anion, MgCO3_Anion,
                    BaSO4_Anion, CaSO4_Anion, SrSO4_Anion, CaF2_Anion]

    Minerals["Cation_Array"] = Cation_Array
    Minerals["Anion_Array"] = Anion_Array

    Minerals["MW_Minerals"] = MW_Minerals
    #Unit Conversions
    Minerals["Cation_per_lb/bbl_water"] = Minerals["Cation_Array"] * 3.78541 * 42 * Minerals["MW_Minerals"] / 453.592
    Minerals["Anion_per_lb/bbl_water"] = Minerals["Anion_Array"] * 3.78541 * 42 * Minerals["MW_Minerals"] / 453.592
    Minerals["Total_Precipitate"] = Minerals["Cation_per_lb/bbl_water"]

    Minerals["Total_Precipitate"] = np.where(Minerals["Cation_per_lb/bbl_water"] > Minerals["Anion_per_lb/bbl_water"],
                                             Minerals["Anion_per_lb/bbl_water"], Minerals["Cation_per_lb/bbl_water"])
    Minerals["Expected_Precipitate_per_1000_bbls"] = Minerals["Total_Precipitate"].apply(lambda x: x*1000 if x >= 0 else 0)

    ####Scaling Indicies
    pCa = -1 * np.log10( Ions.at['Ca_2+', 'meq/L'] /2000 )
    pAlk = -1 * np.log10( Ions.at['HCO3_-', 'Mol/L'] )
    Total_Ion_Conc_Mol_per_L = sum(Ions["Mol/L"]) * 0.5

    if Total_Ion_Conc_Mol_per_L < 1.2:
        K_Stiff_Davis = (2.0222 * 2.71828**((np.log(Total_Ion_Conc_Mol_per_L)+7.5437)**2 / 102.5983) + 
        0.26195 - 0.0097*Temp_C - 0.0002*Temp_C**2)
    else:
        K_Stiff_Davis = (3.652 - 0.1*Total_Ion_Conc_Mol_per_L + 0.26195 - 0.0097*Temp_C - 
        0.0002*Temp_C**2)

    C_Langlier = ((3.25571* 2.71828**(-0.00544*Temp_F)) - 2.2247 - 0.0116*np.log10(TDS_calced)**3 + 
                  0.0905*np.log10(TDS_calced)**2 - 0.1329*np.log10(TDS_calced) + 2.1948)
    X_Skillman =(2.5 * Ions.at['Ca_2+', 'mg/L'] - 1.04*Ions.at['HCO3_-', 'mg/L'] ) * 10**-5
    Langlier = pH - pCa - pAlk - C_Langlier
    Stiff_Davis = pH - pCa - pAlk - K_Stiff_Davis
    Ryznar = pH - 2*Langlier
    
    
    A_var = (np.log10(TDS_calced)-1)/10
    B_var = -13.12*np.log10(Temp_C + 273) + 34.55
    D1_var = np.log10(HCO3)
    D2_var = np.log10(2.5*Ca + 4.1*Mg)  -0.4
    pHs = (9.3+A_var+B_var) - (D1_var + D2_var)
    
    #More accurate Index Calculations
    Langlier_option_2 = pH - pHs
    Ryznar_option_2 = 2*pHs - pH
    Puckorius = 2*pHs - (1.465*np.log10(Ions.at['HCO3_-', 'meq/L']) + 4.54)
    Larson_Skold = (Ions.at['Cl_-', 'meq/L'] + Ions.at['SO4_2-', 'meq/L']) / Ions.at['HCO3_-', 'meq/L'] 
    if Ions.at['SO4_2-', 'meq/L'] < Ions.at['Ca_2+', 'meq/L']:
        Skillman = Ions.at['SO4_2-', 'meq/L'] / (1000 * (0.0025+4*Minerals.at["CaSO4","K_sp_T"] - X_Skillman)**0.5)
    else:
        Skillman = Ions.at['SO4_2-', 'meq/L'] / (1000 * (0.0025+4*Minerals.at["CaSO4","K_sp_T"] - X_Skillman)**0.5)     

    Scale_Indicies_dict = {"Langlier_Saturation_Index" : Langlier_option_2, "Stiff_Davis_Index" : Stiff_Davis,
                     "Ryznar_Index" : Ryznar_option_2, "Puckorius_Index" : Puckorius,"Larson-Skold_Index" : Larson_Skold, 
                           "Skillman_Index" : Skillman}

    Scale_Indicies = pd.DataFrame.from_dict(Scale_Indicies_dict, orient = 'index', columns = ["Indicies"])
    
    Scaling_Calcs.Ions = Ions
    Scaling_Calcs.Minerals = Minerals
    Scaling_Calcs.Scale_Indicies = Scale_Indicies
    

    