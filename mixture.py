from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np 

class mixture(object):
    def __init__(self, component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction):
        #self.component_list = component_list
        #self.mole_fraction = mole_fraction
        self.MW_plus_fraction = MW_plus_fraction
        self.SG_plus_fraction = SG_plus_fraction
        #normalize
        mole_fraction = mole_fraction/np.sum(mole_fraction)

        df = DataFrame(mole_fraction, index = component_list, columns = ['mole_fraction'])
        df = self.remove_zero_components(df)
        self.mole_fraction = df['mole_fraction'].tolist()

        df_pseudo, df_pure = self.separate_pseudocomponents(df)

        all_constants = read_csv('db/constants3.csv', index_col=0)
        self.get_pure_constants(all_constants, df_pure)
        if len(df_pseudo) == 1:
            self.get_pseudo_constants(all_constants, df_pseudo)

        self.component_list = self.molecular_weight.index.to_list()
        self.interaction_params = pd.DataFrame(0, index = self.component_list, columns = self.component_list)
        
        all_interation_params = read_csv('db/interaction_parameters_matrix2.csv', index_col=0)
        self.interaction_params = all_interation_params.loc[self.component_list,self.component_list]



        '''
        constants = all_constants.loc[self.component_list]
        self.antoine_a = constants['antoine_a']
        self.antoine_b = constants['antoine_b']
        self.antoine_c = constants['antoine_c']
        self.molecular_weight = constants['molecular_weight']
        self.Pc = constants['critical_pressure']
        self.Tc = constants['critical_temperature']
        self.w = constants['accentric_factor']
        self.Zc = constants['critical_compressibility']
        self.relative_density = constants['relative_density']
        self.boiling_point = constants['normal_boiling_point']
        self.twu_l = constants['twu_l']
        self.twu_m = constants['twu_m']
        self.twu_n = constants['twu_n']
        all_interation_params = read_csv('db/interaction_parameters_zeros.csv', index_col=0)
        self.interaction_params = all_interation_params.loc[self.component_list,self.component_list]
        '''

    def remove_zero_components(self, df):
        #lazy way of removing zero components
        df = df.replace(0, np.nan)
        df = df.dropna(axis = 0, how = 'any')
        return df


    def separate_pseudocomponents(self,df):
        mask = df.index.str.contains('+', regex = False)
        df_pseudo = df[mask]
        df_pure = df[~mask]
        return df_pseudo, df_pure


    def get_pure_constants(self, all_constants, df_pure):
        component_list = df_pure.index.to_list()
        constants = all_constants.loc[component_list]
        self.antoine_a = constants['antoine_a']
        self.antoine_b = constants['antoine_b']
        self.antoine_c = constants['antoine_c']
        self.molecular_weight = constants['molecular_weight']
        self.Pc = constants['critical_pressure']
        self.Tc = constants['critical_temperature']
        self.w = constants['accentric_factor']
        self.Zc = constants['critical_compressibility']
        self.relative_density = constants['relative_density']
        self.boiling_point = constants['normal_boiling_point']
        self.twu_l = constants['twu_l']
        self.twu_m = constants['twu_m']
        self.twu_n = constants['twu_n']
        self.s = constants['volume_shift']
       
        #self.mole_fraction = df_pure['mole_fraction'].tolist()


    def get_pseudo_constants(self, all_constants, df_pseudo):
        constants = all_constants[all_constants['alkane_carbon_number'] != 0]

        #this will only work for one component
        names = df_pseudo.index.to_list()[0]

        constants = constants.append(pd.Series(name = names))
        constants.at[names,'molecular_weight'] = self.MW_plus_fraction
        constants = constants.sort_values(by = ['molecular_weight'])

        place = np.searchsorted(constants['molecular_weight'],self.MW_plus_fraction)
        
        df_reduced = constants.iloc[place-1:place+2]
        df_reduced = df_reduced.reset_index()
        df_reduced.set_index('molecular_weight',inplace = True)


        Pc = df_reduced['critical_pressure'].interpolate(method = 'values').values[1]
        Tc = df_reduced['critical_temperature'].interpolate(method = 'values').values[1]
        w = df_reduced['accentric_factor'].interpolate(method = 'values').values[1]
        Zc = df_reduced['critical_compressibility'].interpolate(method = 'values').values[1]
        boiling_point = df_reduced['normal_boiling_point'].interpolate(method = 'values').values[1]
        #interpolation doesnt make sense here for twu
        twu_l = df_reduced['twu_l'].interpolate(method = 'values').values[1]
        twu_m = df_reduced['twu_m'].interpolate(method = 'values').values[1]
        twu_n = df_reduced['twu_n'].interpolate(method = 'values').values[1]

        self.CN_plus_frection = df_reduced['alkane_carbon_number'].interpolate(method = 'values').values[1]


        self.molecular_weight = self.molecular_weight.append(pd.Series(self.MW_plus_fraction,index = [names]))
        self.Pc = self.Pc.append(pd.Series(Pc,index = [names]))
        self.Tc = self.Tc.append(pd.Series(Tc,index = [names]))
        self.Zc = self.Zc.append(pd.Series(Zc,index = [names]))
        self.w = self.w.append(pd.Series(w,index = [names]))
        self.relative_density = self.relative_density.append(pd.Series(self.SG_plus_fraction,index = [names]))
        self.boiling_point = self.boiling_point.append(pd.Series(boiling_point,index = [names]))
        self.twu_l = self.twu_l.append(pd.Series(twu_l,index = [names]))
        self.twu_m = self.twu_m.append(pd.Series(twu_m,index = [names]))
        self.twu_n = self.twu_n.append(pd.Series(twu_n,index = [names]))
        
        s_plus_fraction = 1 - (2.258/self.MW_plus_fraction**0.1823)
        self.s = self.s.append(pd.Series(s_plus_fraction,index = [names]))

        #self.mole_fraction.append(df_pseudo['mole_fraction'].to_list()[0])

        '''
        debug_print_list = [self.molecular_weight,
                            self.Pc,
                            self.Tc,
                            self.w,
                            self.relative_density,
                            self.boiling_point,
                            self.twu_l,
                            self.twu_m,
                            self.twu_n]

        for i in debug_print_list:
            print(i)
        '''

if __name__ == "__main__":
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
    mole_fraction = [
                    0.011, 
                    2.301,
                    0.197,
                    3.654,
                    7.368,
                    2.062,
                    7.442,
                    4.32,
                    6.407,
                    11.423,
                    55.175
                    ]

    T = 82.7
    T = (T - 32) * 5 / 9 + 273.15
    P = 128.2
    P = (P + 14.65)/14.5038
    MW_plus_fraction = 215
    SG_plus_fraction = 0.8335


    mix = mixture(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction)
    print(mix.mole_fraction)
    print(mix.CN_plus_frection)
    print(mix.s)

    #print(mix.Pc)

