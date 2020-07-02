import pandas as pd
import numpy as np
import math



def Scale_Calcs_dT(Na, K, Ca, Mg, Ba, Sr, Fe, Mn, Cl, F, HCO3, CO3, SO4, NO3, Temp_C, pH, Silica, H2S):
    ## Part 1 - Ions
    R_const = 8.314
    Temp_K = Temp_C + 273.15
    Temp_F = (Temp_C * 9/5) + 32
    Multiplier = [1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1]
    Ion_Conc_Multiplier = [1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 1]
    Ion_Charge = [1, 1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -2, -2, -1]
    alpha = [0.45, 0.30, 0.60, 0.80, 0.50, 0.50, 0.60, 0.60, 0.30, 0.35, 0.45, 0.45, 0.40, 0.30]
    MW = [22.9898, 39.10, 40.08, 24.31, 137.33, 87.62, 55.85, 54.94, 35.45, 19, 61.02, 60.01, 96.06, 62]
    MW_Minerals = [101.094, 101.094, 116.861, 85.321, 233.386, 136.134, 183.676, 78.074, 172.17, 87.92, 58.44]

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


    ## Part 2 - Minerals:
    R_const = 8.314
    Mineral_names = ["Calcite", "Aragonite", "FeCO3", "MgCO3", "BaSO4", "CaSO4", "SrSO4", "CaF2", 'Gypsum', 'FeS', 'Halite']

    ## put in conditional for this---
    H2S_Mol_L = H2S/(32.65*1000)
    H2S_gamma = 10 ** -((0.509 * (2**2) * (0.5*sum(Ions['Mol/L'].values))**0.5 ) / 
                        ((1+3.29*0.4 * (0.5*sum(Ions['Mol/L'].values))**0.5 )))


    #need excel in availible files
    Energy_df = pd.read_csv('Energy_const.csv')
    Energy_df = Energy_df.set_index('Comp')
    Energy_df.head()

    Mineral_df = pd.DataFrame(index=Mineral_names)
    
    ##Ksp (solubility product constant)
    Calcite = ((np.exp((-((((((Energy_df.at['Ca2+', 'Delta H'] + Energy_df.at['CO3(2-)', 'Delta H'] - 
                                         Energy_df.at['Calcite', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Ca2+', 'Delta G'] + Energy_df.at['CO3(2-)', 'Delta G'] -
                                        Energy_df.at['Calcite', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Ca_2+', 'gamma'] * Ions.at['CO3_2-', 'gamma']))

    Aragonite = ((np.exp((-((((((Energy_df.at['Ca2+', 'Delta H'] + Energy_df.at['CO3(2-)', 'Delta H'] - 
                                         Energy_df.at['Aragonite', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Ca2+', 'Delta G'] + Energy_df.at['CO3(2-)', 'Delta G'] -
                                        Energy_df.at['Aragonite', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Ca_2+', 'gamma'] * Ions.at['CO3_2-', 'gamma']))
    CaSO4 = ((np.exp((-((((((Energy_df.at['Ca2+', 'Delta H'] + Energy_df.at['SO4-2', 'Delta H'] - 
                                         Energy_df.at['CaSO4', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Ca2+', 'Delta G'] + Energy_df.at['SO4-2', 'Delta G'] -
                                        Energy_df.at['CaSO4', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Ca_2+', 'gamma'] * Ions.at['SO4_2-', 'gamma']))
    Gypsum = ((np.exp((-((((((Energy_df.at['Ca2+', 'Delta H'] + 2*Energy_df.at['H2O(l)', 'Delta H'] + Energy_df.at['SO4-2', 'Delta H'] - 
                                         Energy_df.at['CaSO4*2H2O', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Ca2+', 'Delta G'] + 2*Energy_df.at['H2O(l)', 'Delta G'] + Energy_df.at['SO4-2', 'Delta G'] -
                                        Energy_df.at['CaSO4*2H2O', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Ca_2+', 'gamma'] * Ions.at['SO4_2-', 'gamma']))
    BaSO4 = ((np.exp((-((((((Energy_df.at['Ba2+', 'Delta H'] + Energy_df.at['SO4-2', 'Delta H'] - 
                                         Energy_df.at['BaSO4', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Ba2+', 'Delta G'] + Energy_df.at['SO4-2', 'Delta G'] -
                                        Energy_df.at['BaSO4', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Sr_2+', 'gamma'] * Ions.at['SO4_2-', 'gamma']))
    SrSO4 = ((np.exp((-((((((Energy_df.at['Sr2+', 'Delta H'] + Energy_df.at['SO4-2', 'Delta H'] - 
                                         Energy_df.at['SrSO4', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Sr2+', 'Delta G'] + Energy_df.at['SO4-2', 'Delta G'] -
                                        Energy_df.at['SrSO4', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Sr_2+', 'gamma'] * Ions.at['SO4_2-', 'gamma']))
    FeCO3 = ((np.exp((-((((((Energy_df.at['Fe(II)2+', 'Delta H'] + Energy_df.at['CO3(2-)', 'Delta H'] - 
                                         Energy_df.at['FeCO3', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Fe(II)2+', 'Delta G'] + Energy_df.at['CO3(2-)', 'Delta G'] -
                                        Energy_df.at['FeCO3', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Fe_2+', 'gamma'] * Ions.at['CO3_2-', 'gamma']))
    FeS = ((np.exp((-((((((Energy_df.at['Fe(II)2+', 'Delta H'] + Energy_df.at['H2S(g)', 'Delta H'] - 
                                         Energy_df.at['FeS', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Fe(II)2+', 'Delta G'] + Energy_df.at['H2S(g)', 'Delta G'] -
                                        Energy_df.at['FeS', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Fe_2+', 'gamma'] * H2S_gamma))
    Halite = ((np.exp((-((((((Energy_df.at['Na+', 'Delta H'] + Energy_df.at['Cl-', 'Delta H'] - 
                                         Energy_df.at['NaCl', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Na+', 'Delta G'] + Energy_df.at['Cl-', 'Delta G'] -
                                        Energy_df.at['NaCl', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Na_+', 'gamma'] * Ions.at['Cl_-', 'gamma']))
    MgCO3 = ((np.exp((-((((((Energy_df.at['Mg2+', 'Delta H'] + Energy_df.at['CO3(2-)', 'Delta H'] - 
                                         Energy_df.at['MgCO3', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Mg2+', 'Delta G'] + Energy_df.at['CO3(2-)', 'Delta G'] -
                                        Energy_df.at['MgCO3', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Mg_2+', 'gamma'] * Ions.at['CO3_2-', 'gamma']))
    CaF2 = ((np.exp((-((((((Energy_df.at['Ca2+', 'Delta H'] + 2*Energy_df.at['F-', 'Delta H'] - 
                                         Energy_df.at['CaF2', 'Delta H'])*1000)*((1/Temp_K)-(1/298.15)) + 
                                      (((Energy_df.at['Ca2+', 'Delta G'] + 2*Energy_df.at['F-', 'Delta G'] -
                                        Energy_df.at['CaF2', 'Delta G'])*1000)/298.15)))*Temp_K))) / (R_const*Temp_K)))/
                         (Ions.at['Ca_2+', 'gamma'] * Ions.at['F_-', 'gamma']))
    
    ## SI (Saturation Index)
    SI_Calcite = np.log10((Ions.at['Ca_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L']) / Calcite)
    SI_Aragonite = np.log10((Ions.at['Ca_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L']) / Aragonite)
    SI_CaSO4 = np.log10((Ions.at['Ca_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L']) /  CaSO4)
    SI_Gypsum = np.log10((Ions.at['Ca_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L']) / Gypsum)
    SI_BaSO4 = np.log10((Ions.at['Ba_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L']) / BaSO4)
    SI_SrSO4 = np.log10((Ions.at['Sr_2+', 'Mol/L'] * Ions.at['SO4_2-', 'Mol/L']) / SrSO4)
    SI_FeCO3 = np.log10((Ions.at['Fe_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L']) / FeCO3)
    SI_FeS = np.log10((Ions.at['Fe_2+', 'Mol/L'] * H2S_Mol_L) / FeS)
    SI_Halite = np.log10((Ions.at['Na_+', 'Mol/L'] * Ions.at['Cl_-', 'Mol/L']) / Halite)
    SI_MgCO3 = np.log10((Ions.at['Mg_2+', 'Mol/L'] * Ions.at['HCO3_-', 'Mol/L']) / MgCO3)
    SI_CaF2 = np.log10((Ions.at['Ca_2+', 'Mol/L'] * Ions.at['F_-', 'Mol/L']) / CaF2)
    SI_CaF2 = -3.78

    ## Calculate Percipate Mol/L
    cation_Calcite =Ions.at['Ca_2+', 'Mol/L'] - Calcite**0.5
    cation_Aragonite = Ions.at['Ca_2+', 'Mol/L'] - Aragonite**0.5
    cation_CaSO4 = Ions.at['Ca_2+', 'Mol/L'] - CaSO4**0.5
    cation_Gypsum = Ions.at['Ca_2+', 'Mol/L'] - Gypsum**0.5
    cation_BaSO4 = Ions.at['Ba_2+', 'Mol/L'] - BaSO4**0.5
    cation_SrSO4 = Ions.at['Sr_2+', 'Mol/L'] -  SrSO4**0.5
    cation_FeCO3 = Ions.at['Fe_2+', 'Mol/L']  -  FeCO3**0.5
    cation_FeS = Ions.at['Fe_2+', 'Mol/L'] - FeS**0.5
    cation_Halite = Ions.at['Na_+', 'Mol/L']  - Halite**0.5
    cation_MgCO3 =  Ions.at['Mg_2+', 'Mol/L'] - MgCO3**0.5
    cation_CaF2 = Ions.at['Ca_2+', 'Mol/L'] - CaF2**0.5

    anion_Calcite =  Ions.at['HCO3_-', 'Mol/L'] - Calcite**0.5
    anion_Aragonite = Ions.at['HCO3_-', 'Mol/L'] - Aragonite**0.5
    anion_CaSO4 = Ions.at['SO4_2-', 'Mol/L'] - CaSO4**0.5
    anion_Gypsum =  Ions.at['SO4_2-', 'Mol/L'] - Gypsum**0.5
    anion_BaSO4 =Ions.at['SO4_2-', 'Mol/L'] - BaSO4**0.5
    anion_SrSO4 = Ions.at['SO4_2-', 'Mol/L'] -  SrSO4**0.5
    anion_FeCO3 = Ions.at['HCO3_-', 'Mol/L']  -  FeCO3**0.5
    anion_FeS = H2S_Mol_L - FeS**0.5
    anion_Halite = Ions.at['Cl_-', 'Mol/L'] - Halite**0.5
    anion_MgCO3 = Ions.at['HCO3_-', 'Mol/L'] - MgCO3**0.5
    anion_CaF2 = Ions.at['F_-', 'Mol/L'] - CaF2**0.5

    ptb_const = 3.78541*42
    ptb_const2 = 453.592*1000

    MW_Calcite = Ions.at['Ca_2+', 'MW'] + Ions.at['CO3_2-', 'MW']
    MW_Aragonite = Ions.at['Ca_2+', 'MW'] + Ions.at['CO3_2-', 'MW']
    MW_CaSO4 =  Ions.at['Ca_2+', 'MW'] + Ions.at['SO4_2-', 'MW']
    MW_Gypsum = Ions.at['Ca_2+', 'MW'] + Ions.at['SO4_2-', 'MW']
    MW_BaSO4 = Ions.at['Ba_2+', 'MW'] + Ions.at['SO4_2-', 'MW']
    MW_SrSO4 = Ions.at['Sr_2+', 'MW'] + Ions.at['SO4_2-', 'MW'] 
    MW_FeCO3 =  Ions.at['Fe_2+', 'MW'] + Ions.at['CO3_2-', 'MW']
    MW_FeS = Ions.at['Fe_2+', 'Mol/L'] + 34.1
    MW_Halite = Ions.at['Na_+', 'Mol/L'] + Ions.at['Cl_-', 'Mol/L']
    MW_MgCO3 = Ions.at['Mg_2+', 'Mol/L'] + Ions.at['CO3_2-', 'MW']
    MW_CaF2 =Ions.at['Ca_2+', 'Mol/L'] + Ions.at['F_-', 'Mol/L']


    def ptb_results(anion, cation, MW):
        ptb_const = 3.78541*42*1000
        ptb_const2 = 453.592
        if cation < anion:
            ptb = (cation * ptb_const * MW) / ptb_const2
        else:
            ptb = (anion * ptb_const * MW) / ptb_const2
        if ptb < 0:
            ptb = 0
        return ptb

    ptb_Calcite = ptb_results(anion_Calcite, cation_Calcite, MW_Calcite)
    ptb_Aragonite = ptb_results(anion_Aragonite, cation_Aragonite, MW_Aragonite)                                         
    ptb_CaSO4 = ptb_results(anion_CaSO4, cation_CaSO4, MW_CaSO4)                                        
    ptb_Gypsum = ptb_results(anion_Gypsum, cation_Gypsum, MW_Gypsum)
    ptb_BaSO4 = ptb_results(anion_BaSO4, cation_BaSO4, MW_BaSO4)
    ptb_SrSO4 = ptb_results(anion_SrSO4, cation_SrSO4, MW_SrSO4)
    ptb_FeCO3 = ptb_results(anion_FeCO3, cation_FeCO3, MW_FeCO3)
    ptb_FeS = ptb_results(anion_FeS, cation_FeS, MW_FeS)
    ptb_Halite = ptb_results(anion_Halite, cation_Halite, MW_Halite)
    ptb_MgCO3 = ptb_results(anion_MgCO3, cation_MgCO3, MW_MgCO3)
    ptb_CaF2 = ptb_results(anion_CaF2, cation_CaF2, MW_CaF2)
    
    
    ##Scaling Indicies
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
    D2_var = np.log10(2.5*Ca + 4.1*Mg) -0.4
    pHs = (9.3+A_var+B_var) - (D1_var + D2_var)
    
    #More accurate Index Calculations
    Langlier_option_2 = pH - pHs
    Ryznar_option_2 = 2*pHs - pH
    Puckorius = 2*pHs - (1.465*np.log10(Ions.at['HCO3_-', 'meq/L']) + 4.54)
    Larson_Skold = (Ions.at['Cl_-', 'meq/L'] + Ions.at['SO4_2-', 'meq/L']) / Ions.at['HCO3_-', 'meq/L'] 
    if Ions.at['SO4_2-', 'meq/L'] < Ions.at['Ca_2+', 'meq/L']:
        Skillman = Ions.at['SO4_2-', 'meq/L'] / (1000 * (0.0025+4 * Calcite - X_Skillman)**0.5)
    else:
        Skillman = Ions.at['SO4_2-', 'meq/L'] / (1000 * (0.0025+4 * Calcite - X_Skillman)**0.5)     

    Mineral_names = ["Calcite", "Aragonite", "CaSO4", "Gypsum", "BaSO4",
                     "SrSO4", "FeCO3", "FeS", 'Halite', 'MgCO3', 'CaF2']  
    ptb_data = [ptb_Calcite, ptb_Aragonite, ptb_CaSO4, ptb_Gypsum, ptb_BaSO4,
               ptb_SrSO4, ptb_FeCO3, ptb_FeS, ptb_Halite, ptb_MgCO3, ptb_CaF2]   
    SI_data = [SI_Calcite, SI_Aragonite, SI_CaSO4, SI_Gypsum, SI_BaSO4,
                       SI_SrSO4, SI_FeCO3, SI_FeS, SI_Halite, SI_MgCO3, SI_CaF2]
    Scale_index = [Langlier_option_2, Stiff_Davis, Ryznar_option_2, Puckorius, Larson_Skold,Skillman]
    Scale_index_names = ["Langlier_Saturation_Index", "Stiff_Davis_Index", "Ryznar_Index", 
                         "Puckorius_Index", "Larson-Skold_Index", "Skillman_Index"]
    
    
    return Ions, Mineral_names, ptb_data, SI_data, Scale_index, Scale_index_names
    