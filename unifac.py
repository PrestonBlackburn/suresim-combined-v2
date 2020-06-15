from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np


class unifac():
    def __init__(self, component_list, CN_plus_fraction):
        pseuso_list = [s for s in component_list if '+' in s]
        pure_list = [s for s in component_list if '+' not in s]

        self.component_list = pure_list

        all_raw_groups = read_csv('db/unifac_component.csv', index_col=0)
        raw_groups = all_raw_groups.loc[self.component_list]

        pseudo_dict1 = {pseuso_list[0]: ['CH3', 2]}
        pseudo_dict2 = {pseuso_list[0]: ['CH2', CN_plus_fraction-2]}
        pseudo_df1 = DataFrame.from_dict(pseudo_dict1, orient = 'index', columns = ['group', 'count'])
        pseudo_df2 = DataFrame.from_dict(pseudo_dict2, orient = 'index', columns = ['group', 'count']) 
        pseudo_df = pseudo_df1.append(pseudo_df2)

        raw_groups = raw_groups.append(pseudo_df)

        #extraxt all unigue subgroups to use as rows
        self.groups = raw_groups.group.unique()

        #initialize df of zeros with components as columns, groups as rows
        self.df = pd.DataFrame(0 ,columns = self.component_list, index = self.groups)

        #iterate through list from db and place into matrix
        for index, row in raw_groups.iterrows():
            self.df.at[row['group'],index] = row['count']
        
        #get q from the subgroup table
        all_sub_groups = read_csv('db/unifac_subgroup.csv', index_col=1)
        self.sub_groups = all_sub_groups.loc[self.groups]

        #get main group id the extract unique values
        mainid = self.sub_groups['maingroup_id']
        unique_mainid = mainid.unique().tolist()

        all_interaction = read_csv('db/unifac_interaction.csv')
        #query for only needed interaction params
        A = all_interaction.query('i == @unique_mainid').query('j == @unique_mainid')

        #generate matrix, will assign zeros after
        self.Aij = pd.DataFrame(columns = mainid.values, index = mainid.values)

        for row in A.itertuples():
            self.Aij.loc[row.i,row.j] = row.Aij
            self.Aij.loc[row.j,row.i] = row.Aji
        self.Aij.fillna(0, inplace=True)

        #re-name rows and colums to group names to keep my sanity
        self.Aij.columns = self.groups.tolist()
        self.Aij.index = self.groups.tolist()



    def gibbs_res(self,mole_fraction,T, DebugPrint = False):
        #normalize the matrix and add mixture column
        mixture = self.df*mole_fraction
        self.df['mixture'] = mixture.sum(axis = 1)
        Xsum = self.df.sum(axis=0)
        Xi = self.df/Xsum

        Qi = self.sub_groups['Q']
        XiQi = Xi.multiply(Qi, axis = 0)
        sumXiQi = XiQi.sum(axis = 0)

        theta = XiQi/sumXiQi

        tau = np.exp(-1*self.Aij/T)

        sumTheta_i_Psi_ij = np.dot(tau.transpose(),theta)

        #labeling things again to keep sanity, also labels make division easier in next step
        sumTheta_i_Psi_ij = pd.DataFrame(sumTheta_i_Psi_ij, columns = theta.columns, index = theta.index)

        Theta_j_div_sumTheta_i_Psi_ij = theta/sumTheta_i_Psi_ij

        sum_jk = np.dot(tau, Theta_j_div_sumTheta_i_Psi_ij)
        sum_jk = pd.DataFrame(sum_jk, columns = theta.columns, index = theta.index)

        ln_Gamma = np.log(sumTheta_i_Psi_ij)
        ln_Gamma = np.subtract(1,ln_Gamma)
        ln_Gamma = np.subtract(ln_Gamma,sum_jk)
        ln_Gamma = ln_Gamma.multiply(Qi, axis = 0)

        ln_gamma_res = self.df.multiply(ln_Gamma['mixture'], axis = 0)
        ln_gamma_res = ln_gamma_res.sum(axis = 0)
        ln_gamma_res = ln_gamma_res.drop(['mixture'])

        # messed up somewhere :( just trying to repair lazily
        if 'mixture' in self.df.columns:
            self.df = self.df.drop(['mixture'], axis = 1)

        Gibbs_res = ln_gamma_res * mole_fraction
        Gibbs_res = Gibbs_res.sum(axis = 0)
        # * 8.314 * T

        if DebugPrint:
            print(Xi)
            print(self.sub_groups)
            print(XiQi)
            print(sumXiQi)
            print(theta)
            print(self.Aij)
            print(tau)
            print(sumTheta_i_Psi_ij)
            print(Theta_j_div_sumTheta_i_Psi_ij)
            print(sum_jk)
            print(ln_Gamma)
            print(ln_gamma_res)
            print("Gibbs res:",Gibbs_res)


        return(Gibbs_res)


if __name__ == "__main__":

    T = 80.37 + 273.15

    component_list = [
        'Nitrogen', 
        'Carbon Dioxide', 
        'Methane', 
        'Ethane', 
        'Propane', 
        'Isobutane', 
        'n-Butane', 
        'Isopentane',
        'n-Pentane', 
        'n-Hexane', 
        'n-Heptane', 
        'n-Octane', 
        'n-Nonane', 
        'n-Decane+', 
        'Benzene', 
        'Toluene', 
        'Xylenes', 
        'Cyclohexane' 
    ]
    mole_fraction = [0.0, 0.0002219099384077, 0.0161323268823673, 0.034306066430663695, 0.061376714536037894, 0.017411880841312198, 0.06998357896652878, 0.029745615439193095, 0.056242562980590494, 0.07744530004039975, 0.11103942723847349, 0.09260349213897238, 0.05939777311038963, 0.3409167965622543, 0.003109978591283919, 0.00673044828123712, 0.01387876828400669, 0.009457359737881428]



    uni = unifac(component_list,15.183949825386643)
    gam = uni.gibbs_res(mole_fraction,T, DebugPrint=True)
    print(gam)


