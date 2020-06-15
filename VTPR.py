from pandas import read_csv, DataFrame
from numpy import outer, multiply, power, subtract, sum as npsum, dot, sign, nan, cos, pi, arctan, exp, sqrt, log, roots, add
from cmath import log as clog
import numpy
import math
import os
import time

from scipy.optimize import fsolve, newton, fminbound, brenth
import matplotlib.pyplot as plt

from unifac import unifac
from mixture import mixture


# ToDo: add dew options and wobbe: examples on Lab/kirsten/COC tags/Thermound


class eos(mixture):
    # anytime T,P,X is called should be a method to allow fugacity calcs
    def __init__(self, component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction):
        mixture.__init__(self, component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction)
        
        # I dont think R really belongs here
        self.R = 8.3145
        self.Kappa = 0.37464 + 1.54226*self.w - 0.26992*(self.w.pow(2))
        
    def mixture_density(self, P, T):
        Z = min(self.Z)
        R = self.R
        V = (R*T*Z)/P
        density = sum(self.molecular_weight * self.mole_fraction) / V
        return density *100

    def volume_translation(self, P, T):
        pass

    def bubble_func(self):
        pass


class VTPR(eos):
    # anytime T,P,X is called should be a method to allow fugacity calcs

    def __init__(self, component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T):
        eos.__init__(self, component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction)
        self.unifac_model = unifac(self.component_list, self.CN_plus_frection)
        #self.Z = self._solve_cubic(mole_fraction, P, T)
        #dummy, self.ZZ = self.fugacity_coeffecient(mole_fraction, P, T,'liquid')
        self.T = T
        self.P = P
        #self.c = 0.252 * self.R * self.Tc / self.Pc * (1.5448*self.Zc - 0.4024)



    def _alpha_soave(self, T):
        # not sure if kappa is only related to soave alpha functio
        alpha = (1 + self.Kappa * (1 - (T/self.Tc).pow(0.5))).pow(2)
        return alpha


    def _Twu91(self,T):
        Tr = T/self.Tc
        alpha = Tr.pow(self.twu_n*(self.twu_m-1))*exp(self.twu_l*(1-Tr.pow(self.twu_m*self.twu_n)))
        return alpha


    def _Twu95(self, T):
        Tr = T/self.Tc
        L0 = []
        M0 = []
        N0 = []
        L1 = []
        M1 = []
        N1 = []
        for tr in Tr:
            if tr <= 1:
                L0.append(0.125283)
                M0.append(0.911807)
                N0.append(1.948150)

                L1.append(0.511614)
                M1.append(0.784054)
                N1.append(2.812520)
            else:
                L0.append(0.401219)
                M0.append(4.963070)
                N0.append(-0.2)

                L1.append(0.024955)
                M1.append(1.248089)
                N1.append(-8.0)

        L0 = numpy.asarray(L0)
        M0 = numpy.asarray(M0)
        N0 = numpy.asarray(N0)
        L1 = numpy.asarray(L1)
        M1 = numpy.asarray(M0)
        N1 = numpy.asarray(N0)
        a0 = Tr.pow(N0*(M0-1))*exp(L0*(1-Tr.pow(M0*N0)))
        a1 = Tr.pow(N1*(M1-1))*exp(L1*(1-Tr.pow(M1*N1)))
        alpha = a0 + self.w *(a1 - a0)
        return alpha

    def _Twu_hybrid(self, T):
        # doing this lazy way until verfied
        Tr = T/self.Tc
        L0 = []
        M0 = []
        N0 = []
        L1 = []
        M1 = []
        N1 = []
        for index, tr in Tr.iteritems():
            if tr <= 1:
                L0.append(self.twu_l[index])
                M0.append(self.twu_m[index])
                N0.append(self.twu_n[index])

                L1.append(self.twu_l[index])
                M1.append(self.twu_m[index])
                N1.append(self.twu_n[index])

            else:
                L0.append(0.401219)
                M0.append(4.963070)
                N0.append(-0.2)

                L1.append(0.024955)
                M1.append(1.248089)
                N1.append(-8.0)

        L0 = numpy.asarray(L0)
        M0 = numpy.asarray(M0)
        N0 = numpy.asarray(N0)
        L1 = numpy.asarray(L1)
        M1 = numpy.asarray(M0)
        N1 = numpy.asarray(N0)
        a0 = Tr.pow(N0*(M0-1))*exp(L0*(1-Tr.pow(M0*N0)))
        a1 = Tr.pow(N1*(M1-1))*exp(L1*(1-Tr.pow(M1*N1)))
        alpha = a0 + self.w *(a1 - a0)
        return alpha


    def _van_der_walls_mixing(self, mole_fraction, P, T):
        alpha = self._Twu91(T)
        attraction = 0.45724 * alpha * (self.R * self.Tc).pow(2)/(self.Pc * 100000)
        cohesion = 0.07780 * self.R * self.Tc / (self.Pc * 100000)
        # aij = [(ai.aj)0.5(1 - kij)] = aji
        mixture = outer(attraction, attraction)
        mixture = power(mixture,0.5)
        mixture = multiply(mixture, subtract(1,self.interaction_params))
        a_ij = mixture * P * 100000 / (self.R * T) ** 2
        b = cohesion * P * 100000 / (self.R * T)
        A = npsum(npsum(multiply(a_ij, outer(mole_fraction, mole_fraction))))
        B = npsum(b*mole_fraction)
        return A, B, a_ij, b


    def _mod_van_der_walls_mixing(self, mole_fraction, P, T):
        alpha = self._Twu91(T)
        attraction = 0.45724 * alpha * (self.R * self.Tc).pow(2)/(self.Pc * 100000)
        cohesion = 0.07780 * self.R * self.Tc / (self.Pc * 100000)
        # aij = [(ai.aj)0.5(1 - kij)] = aji
        mixture = outer(attraction, attraction)
        mixture = power(mixture,0.5)
        mixture = multiply(mixture, subtract(1,self.interaction_params))
        a_ij = mixture * P * 100000 / (self.R * T) ** 2
        b = cohesion * P * 100000 / (self.R * T)

        bi = numpy.asarray(b.pow(3/4))
        b_ij = (add.outer(bi,bi)/2)
        b_ij = b_ij**(4/3)

        A = npsum(npsum(multiply(a_ij, outer(mole_fraction, mole_fraction))))
        B = npsum(npsum(multiply(b_ij, outer(mole_fraction, mole_fraction))))
        return A, B, a_ij, b


    def MHV1(self,mole_fraction,P,T):
        g_res = self.unifac_model.gibbs_res(mole_fraction = mole_fraction,T = T, DebugPrint = False)
        
        alpha = self._Twu91(T)
        attraction = 0.45724 * alpha * (self.R * self.Tc).pow(2)/(self.Pc * 100000)
        cohesion = 0.07780 * self.R * self.Tc / (self.Pc * 100000)
        # aij = [(ai.aj)0.5(1 - kij)] = aji
        mixture = outer(attraction, attraction)
        mixture = power(mixture,0.5)
        mixture = multiply(mixture, subtract(1,self.interaction_params))
        a_ij = mixture * P * 100000 / (self.R * T) ** 2
        b = cohesion * P * 100000 / (self.R * T)
        c = self.s * cohesion

        bi = numpy.asarray(b.pow(3/4))
        b_ij = (add.outer(bi,bi)/2)
        b_ij = b_ij**(4/3)
        
        B = npsum(npsum(multiply(b_ij, outer(mole_fraction, mole_fraction))))
        A = B* (npsum(attraction/(cohesion * self.R *T) * mole_fraction)- g_res/0.53087)
        C = npsum(c*mole_fraction)
        return A, B, a_ij, b, C


    def _solve_cubic(self, mole_fraction, P, T, A, B):      
        C2 = B - 1
        C1 = A-3*B**2-2*B
        C0 = -1*A*B+B**2+B**3
        #new method
        poly = [1,C2,C1,C0]
        Z2 = roots(poly)
        Z2 =Z2[numpy.isreal(Z2)]
        return Z2.real


    def fugacity_coeffecient(self, mole_fraction, P, T, liquid_or_vapor):
        A, B, a_ij, b, C = self.MHV1(mole_fraction, P, T)
        Z = self._solve_cubic(mole_fraction, P, T, A, B)
        if liquid_or_vapor == 'liquid':
            Z = min(Z)
        elif liquid_or_vapor == 'vapor':
            Z = max(Z)

        log_phi = A/(2*sqrt(2)*B)
        log_phi *= (b/B) - 2*(dot(a_ij, mole_fraction)/A)
        log_phi *= log((Z + (1 + sqrt(2))*B) / (Z + (1 - sqrt(2))*B))
        log_phi += b/B*(Z - 1) - log(Z - B)
        phi = exp(log_phi)
        if B == 7987987:
            print("B = ",B)
            print(mole_fraction)
            print(liquid_or_vapor)
            time.sleep(2)
        return phi , Z


    def fugacity_coeffecient_VT(self, mole_fraction, P, T, liquid_or_vapor):
        A, B, a_ij, b, C = self.MHV1(mole_fraction, P, T)
        Z = self._solve_cubic(mole_fraction, P, T, A, B)
        if liquid_or_vapor == 'liquid':
            Z = min(Z)
        elif liquid_or_vapor == 'vapor':
            Z = max(Z)

        log_phi = A/(2*sqrt(2)*B)
        log_phi *= (b/B) - 2*(dot(a_ij, mole_fraction)/A)
        log_phi *= log((Z + (1 + sqrt(2))*B) / (Z + (1 - sqrt(2))*B))
        log_phi += b/B*(Z - 1) - log(Z - B)
        phi = exp(log_phi)
        if B == 7987987:
            print("B = ",B)
            print(mole_fraction)
            print(liquid_or_vapor)
            time.sleep(2)
        return phi, Z, C

    

    def initial_P_bubble(self):
        #P_ant = exp(self.antoine_a - (self.antoine_b/(self.T + self.antoine_c)))*self.mole_fraction/750.062
        #print(npsum(P_ant))
        #P_bubble_est = 0.8*npsum(self.mole_fraction*self.Pc)
        #print(P_bubble_est)
        P_bubble_est = 0.7*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        #print(P_bubble_est)

        T_low = T_high = 0

        for i in range(30):
            #print(T_low,T_high)
            K = self.Pc/P_bubble_est * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
            #K = self.T/self.Tc(1/(1/P_bubble_est - 1/P_ant)*(1/self.Pc - 1/P_ant))
            y = npsum(self.mole_fraction*K)
            #print(y)
            if y > 1:
                T_low = P_bubble_est
                y_low = y - 1
                T_new = P_bubble_est * 1.1
            elif y < 1:
                T_high = P_bubble_est
                y_high = y - 1
                T_new = P_bubble_est / 1.1
            else:
                return P_bubble_est
            
            if T_low * T_high > 0:
                T_new = (y_high*T_low - y_low*T_high)/(y_high - y_low)

            if abs(P_bubble_est - T_new) < 0.001:
                return P_bubble_est

            if abs(y - 1) < 0.00001:
                return P_bubble_est

            P_bubble_est = T_new

    
    def restricted_mod_bubble_P(self):
        P_bubble_old = 0.6*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        print(P_bubble_old)
        restriction = numpy.inf
        rest_iter = 0
        while restriction > 100 and rest_iter <10:
            rest_iter+=1
            P_bubble = P_bubble_old*0.9
            lap = 1
            K_old = self.Pc/P_bubble * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
            main_iter1 = 0
            while True:
                x= self.mole_fraction*K_old
                i = 0
                while True:
                    i += 1
                    x_old = x
                    phiL, ZL = self.fugacity_coeffecient(self.mole_fraction,P_bubble,self.T,'liquid')
                    phiV, ZV = self.fugacity_coeffecient(x_old,P_bubble,self.T,'vapor')
                    K_PR = phiL/phiV
                    x = self.mole_fraction*K_PR
                    x_sum = npsum(x)
                    x = x/x_sum
                    diff = x/x_old - 1
                    if all(diff) < 0.001 or i>50:
                        break

                if lap == 1:
                    x_low = x_high = x_sum - 1
                    T_low = T_high = P_bubble

                if x_sum < 1:
                    T_high = P_bubble
                    x_high = x_sum - 1
                    P_bubble = P_bubble / 1.202
                elif x_sum > 1:
                    T_low = P_bubble
                    x_low = x_sum - 1
                    P_bubble = P_bubble * 1.201
                
                main_iter1 += 1
                #print(T_low, T_high, lap)
                if x_low * x_high > 0 or x_low == 0 or x_high == 0:
                    lap += 1
                elif main_iter1 > 50:
                    print("bracketing no convergence")
                    break
                else:
                    break
            main_iter = 0    
            while True:
                main_iter += 1
                P_new = (T_low + T_high)/2
                x= self.mole_fraction*K_old
                    #x = x/npsum(x)
                    #might need to normalize this ^
                i = 0
                while True:
                    i +=1
                    x_old = x
                    phiL, ZL = self.fugacity_coeffecient(self.mole_fraction,P_new,self.T,'liquid')
                    phiV, ZV = self.fugacity_coeffecient(x_old,P_new,self.T,'vapor')
                    K_PR = phiL/phiV
                    x = self.mole_fraction*K_PR
                    x_sum = npsum(x)
                    x = x/x_sum
                    diff = x/x_old - 1
                    if all(diff) < 0.001 or i > 50:
                        break
                
                x_new = x_sum - 1
                
                if x_low * x_new > 0:
                    x_low = x_new
                    T_low = P_new
                else:
                    x_high = x_new
                    T_high = P_new

                if T_high - T_low < 0.01:
                    restriction = abs(P_bubble_old-P_new)/(P_bubble_old/2 + P_new/2)*100
                    if restriction < 100:
                        return P_new
                
                if main_iter > 50:
                    print("no bubble convergence in bracket")
                    return None


    def mod_bubble_P(self):
        P_bubble = self.initial_P_bubble()
        print("Initial bubble", P_bubble)
        #P_bubble = 0.5*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        #P_ant = exp(self.antoine_a - (self.antoine_b/(self.T + self.antoine_c)))*self.mole_fraction/750.062
        #P_bubble = 0.8*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        #print("estimated bubble P", P_bubble)
        lap = 1
        K_old = self.Pc/P_bubble * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
        #K_old = self.T/self.Tc*exp(1/(1/P_bubble - 1/P_ant)*(1/self.Pc - 1/P_ant))
        main_iter1 = 0
        while True:
            x= self.mole_fraction*K_old
            #x = x/npsum(x)
            #might need to normalize this ^
            i = 0
            while True:
                i += 1
                x_old = x
                phiL, ZL = self.fugacity_coeffecient(self.mole_fraction,P_bubble,self.T,'liquid')
                phiV, ZV = self.fugacity_coeffecient(x_old,P_bubble,self.T,'vapor')
                K_PR = phiL/phiV
                x = self.mole_fraction*K_PR
                x_sum = npsum(x)
                x = x/x_sum
                diff = x/x_old - 1
                if all(diff) < 0.001 or i>20:
                    break

            if lap == 1:
                x_low = x_high = x_sum - 1
                T_low = T_high = P_bubble

            if x_sum < 1:
                T_high = P_bubble
                x_high = x_sum - 1
                P_bubble = P_bubble / 1.202
            elif x_sum > 1:
                T_low = P_bubble
                x_low = x_sum - 1
                P_bubble = P_bubble * 1.201
            
            main_iter1 += 1
            #print(T_low, T_high, lap)
            if x_low * x_high > 0 or x_low == 0 or x_high == 0:
                lap += 1
            elif main_iter1 > 50:
                print("bracketing no convergence")
                break
            else:
                break
        main_iter = 0    
        while True:
            main_iter += 1
            P_new = (T_low + T_high)/2
            x= self.mole_fraction*K_old
                #x = x/npsum(x)
                #might need to normalize this ^
            i = 0
            while True:
                i +=1
                x_old = x
                phiL, ZL = self.fugacity_coeffecient(self.mole_fraction,P_new,self.T,'liquid')
                phiV, ZV = self.fugacity_coeffecient(x_old,P_new,self.T,'vapor')
                K_PR = phiL/phiV
                x = self.mole_fraction*K_PR
                x_sum = npsum(x)
                x = x/x_sum
                diff = x/x_old - 1
                if all(diff) < 0.001 or i > 10:
                    break
            
            x_new = x_sum - 1
            
            if x_low * x_new > 0:
                x_low = x_new
                T_low = P_new
            else:
                x_high = x_new
                T_high = P_new

            if T_high - T_low < 0.01:
                return P_new
            
            if main_iter > 50:
                print("no bubble convergence in bracket")
                return None

    def mod_Dew_P(self):
        #P_bubble = self.initial_P_bubble()
        #print("Initial bubble", P_bubble)
        P_bubble = 0.5*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        #P_ant = exp(self.antoine_a - (self.antoine_b/(self.T + self.antoine_c)))*self.mole_fraction/750.062
        #P_bubble = 0.8*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        #print("estimated bubble P", P_bubble)
        lap = 1
        K_old = self.Pc/P_bubble * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
        #K_old = self.T/self.Tc*exp(1/(1/P_bubble - 1/P_ant)*(1/self.Pc - 1/P_ant))
        main_iter1 = 0
        while True:
            x= self.mole_fraction/K_old
            #x = x/npsum(x)
            #might need to normalize this ^
            i = 0
            while True:
                i += 1
                x_old = x
                phiV, ZV = self.fugacity_coeffecient(self.mole_fraction,P_bubble,self.T,'vapor')
                phiL, ZL = self.fugacity_coeffecient(x_old,P_bubble,self.T,'liquid')
                K_PR = phiL/phiV
                x = self.mole_fraction/K_PR
                x_sum = npsum(x)
                x = x/x_sum
                diff = x/x_old - 1
                if all(diff) < 0.001 or i>20:
                    break

            if lap == 1:
                x_low = x_high = x_sum - 1
                T_low = T_high = P_bubble

            if x_sum < 1:
                T_high = P_bubble
                x_high = x_sum - 1
                P_bubble = P_bubble / 1.202
            elif x_sum > 1:
                T_low = P_bubble
                x_low = x_sum - 1
                P_bubble = P_bubble * 1.201
            
            main_iter1 += 1
            #print(T_low, T_high, lap)
            if x_low * x_high > 0 or x_low == 0 or x_high == 0:
                lap += 1
            elif main_iter1 > 50:
                print("bracketing no convergence")
                break
            else:
                break
        main_iter = 0    
        while True:
            main_iter += 1
            P_new = (T_low + T_high)/2
            x= self.mole_fraction/K_old
                #x = x/npsum(x)
                #might need to normalize this ^
            i = 0
            while True:
                i +=1
                x_old = x
                phiV, ZV = self.fugacity_coeffecient(self.mole_fraction,P_new,self.T,'vapor')
                phiL, ZL = self.fugacity_coeffecient(x_old,P_new,self.T,'liquid')
                K_PR = phiL/phiV
                x = self.mole_fraction/K_PR
                x_sum = npsum(x)
                x = x/x_sum
                diff = x/x_old - 1
                if all(diff) < 0.001 or i > 10:
                    break
            
            x_new = x_sum - 1
            
            if x_low * x_new > 0:
                x_low = x_new
                T_low = P_new
            else:
                x_high = x_new
                T_high = P_new

            if T_high - T_low < 0.01:
                return P_new
            
            if main_iter > 50:
                print("no bubble convergence in bracket")
                return None
    

    def bubble_pressure(self, graph = False):
        def bubble(P):
            y = self.mole_fraction
            phiL, ZL = self.fugacity_coeffecient(y,P,self.T,'liquid')
            fL = phiL*y*P
            K_est = self.Pc/P * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
            x = y*K_est
            x = x/npsum(x)
            phiV, ZV = self.fugacity_coeffecient(x,P,self.T,'vapor')
            fV = phiV*x*P
            for i in range(50):
                x = fL/fV*x
                phiV, ZV = self.fugacity_coeffecient(x,P,self.T,'vapor')
                fV = phiV*x*P   
            sum2 = npsum(x)
            return sum2 - 1
        P_bubble_est = npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        root = newton(bubble,0.6*P_bubble_est)
        
        if graph:
            x_plot = numpy.linspace(start = 0.5*P_bubble_est, stop = P_bubble_est, num = 40)
            y_func = numpy.vectorize(bubble)
            y_plot = y_func(x_plot)
            plt.plot(x_plot,y_plot)
            plt.axhline(y=0, color='g', linestyle='--', ms = 1)
            #plt.plot(root, 0, marker='o', markersize=4, color='r')
            #plt.plot(P_bubble_est, 0, marker='o', markersize=2, color='cyan')
            plt.savefig("graphs/bubbleP.png")
            plt.close()
        
        return root           
    
    def dew_pressure(self, graph = False):
        def dew(P):
            y = self.mole_fraction
            phiV, ZV = self.fugacity_coeffecient(y,P,self.T,'vapor')
            fV = phiV*y*P
            K_est = self.Pc/P * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
            x = y/K_est
            x = x/npsum(x)
            phiL, ZL = self.fugacity_coeffecient(x,P,self.T,'liquid')
            fL = phiL*x*P
            for i in range(50):
                x = fV/fL*x
                print(x)
                phiL, ZL = self.fugacity_coeffecient(x,P,self.T,'liquid')
                print(phiL)
                fL = phiL*x*P
            sum2 = npsum(x)
            return sum2 - 1
        P_dew_est = 0.4*npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        root = newton(dew,P_dew_est)
        
        if graph:
            x_plot = numpy.linspace(start = 1, stop = 30, num = 40)
            y_func = numpy.vectorize(dew)
            y_plot = y_func(x_plot)
            plt.plot(x_plot,y_plot)
            plt.axhline(y=0, color='g', linestyle='--', ms = 1)
            #plt.plot(root, 0, marker='o', markersize=4, color='r')
            #plt.plot(P_dew_est, 0, marker='o', markersize=2, color='cyan')
            plt.savefig("graphs/dewP.png")
            plt.close()
        
        return root

    def dew_pressure2(self, graph = False):
        def dew(P):
            y = self.mole_fraction
            phiV = self.fugacity_coeffecient(y,P,self.T,'vapor')
            fV = phiV*y*P
            K_est = self.Pc/P * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
            x = y/K_est
            x = x/npsum(x)
            phiL = self.fugacity_coeffecient(x,P,self.T,'liquid')
            fL = phiL*x*P
            for i in range(50):
                x = fV/fL*x
                #x = x/npsum(x)
                phiL = self.fugacity_coeffecient(x,P,self.T,'liquid')
                fL = phiL*x*P
            sum2 = npsum(x)
            return abs(sum2 - 1)
        P_dew_est = npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        root = fminbound(dew,0.2*P_dew_est, 0.5*P_dew_est)
        #root = newton(dew,0.2*P_dew_est)
        #root = 1
        
        if graph:
            x_plot = numpy.linspace(start = 0.2*P_dew_est, stop = 0.6*P_dew_est, num = 30)
            y_func = numpy.vectorize(dew)
            y_plot = y_func(x_plot)
            plt.plot(x_plot,y_plot)
            plt.axhline(y=0, color='g', linestyle='--', ms = 1)
            #plt.plot(root, 0, marker='o', markersize=4, color='r')
            #plt.plot(P_dew_est, 0, marker='o', markersize=2, color='cyan')
            plt.savefig("graphs/dewP2.png")
            plt.close()
        
        return root

    def bubble_pressure2(self, graph = False):
        def bubble(P):
            y = self.mole_fraction
            phiL, ZL = self.fugacity_coeffecient(y,P,self.T,'liquid')
            fL = phiL*y*P
            K_est = self.Pc/P * exp(5.37*(1 - self.w)*(1 - self.Tc/self.T))
            x = y*K_est
            x = x/npsum(x)
            phiV, ZV = self.fugacity_coeffecient(x,P,self.T,'vapor')
            fV = phiV*x*P
            for i in range(50):
                x = fL/fV*x
                phiV, ZV = self.fugacity_coeffecient(x,P,self.T,'vapor')
                fV = phiV*x*P   
            sum2 = npsum(x)
            return abs(sum2 - 1)
        P_bubble_est = npsum(self.mole_fraction*exp(log(self.Pc) + log(10)*(7/3)*(1 + self.w)*(1-self.Tc/self.T)))
        root = fminbound(bubble,0.5*P_bubble_est,P_bubble_est)
        
        if graph:
            x_plot = numpy.linspace(start = 0.5*P_bubble_est, stop = P_bubble_est, num = 40)
            y_func = numpy.vectorize(bubble)
            y_plot = y_func(x_plot)
            plt.plot(x_plot,y_plot)
            plt.axhline(y=0, color='g', linestyle='--', ms = 1)
            #plt.plot(root, 0, marker='o', markersize=4, color='r')
            #plt.plot(P_bubble_est, 0, marker='o', markersize=2, color='cyan')
            plt.savefig("graphs/bubbleP.png")
            plt.close()
        
        return root

    def pt_flash(self, P, T):
        Kold = exp(log(self.Pc/P) + log(10)*(7/3)*(1 + self.w)*(1 - self.Tc/T))
        found = False
        max_iter = 10
        i = 0
        while not found and i < max_iter:
            i = i+1
            K, diff = self.calculate_K(Kold,P,T)
            if diff <= 1.0e-16:
                found = True
            Kold = K


    def calculate_K(self,K0, P, T):
        def funct(V):
            F = self.mole_fraction*(K0 - 1)/(V*(K0-1)+1)
            if V>0:
                return F.sum()
            else:
                return np.nan
        (root,root_res) = brenth(funct, 0.1e-8, 1,disp = False, full_output = True)

        
        x = self.mole_fraction/(root*(K0-1)+1)
        self.x = x/npsum(x)
        y = x*K0
        self.y = y/sum(y)

        phiL, self.ZL = self.fugacity_coeffecient(x,P,T,'liquid')
        phiV, self.ZV = self.fugacity_coeffecient(y,P,T,'vapor')

        Knew = phiL/phiV
        diff = abs(npsum(Knew) - npsum(K0))
        K = Knew

        self.V = root
        self.L = 1 - root

        return K , diff

    def shrinkage(self, bubble_point_force = True):
        Tx = (60 - 32) * 5 / 9 + 273.15
        Px = 1.0155
        self.pt_flash(Px, Tx)
        dummy, Zx, Cx = self.fugacity_coeffecient_VT(self.x.values.tolist(), Px, Tx,'liquid')
        
        if bubble_point_force:
            bubble = self.bubble_pressure2()
            dummy, Zz, Cz = self.fugacity_coeffecient_VT(self.mole_fraction, bubble, self.T,'liquid')
            Vin = Zz * self.T / bubble - Cz
        else:
            dummy, Zz, Cz = self.fugacity_coeffecient_VT(self.mole_fraction, self.P, self.T,'liquid')
            Vin = Zz * self.T / self.P - Cz 
        Vout = self.L * (Zx * Tx / Px - Cx)
        shrinkage = Vout / Vin

        return shrinkage



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
                    0.042, 
                    24.178,
                    1.178,
                    12.418,
                    9.459,
                    3.398,
                    5.968,
                    3.725,
                    3.414,
                    5.72,
                    30.5
                    ]

    T = 119
    T = (T - 32) * 5 / 9 + 273.15
    P = 128
    P = (P + 14.65)/14.5038
    MW_plus_fraction = 160
    SG_plus_fraction = 0.7768

    eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
    shrinkage = eos.shrinkage(bubble_point_force=False)
    print(shrinkage)


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

    eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)
    shrinkage = eos.shrinkage()
    print(shrinkage)

'''
    

'''

    Tx = (60 - 32) * 5 / 9 + 273.15
    Px = 1.0155
    eos.pt_flash(Px, Tx)
    print(eos.y)
    print(eos.x)
    print(eos.ZV)
    print(eos.ZL) 
    print(eos.x.values.tolist())
    dummy, Zx, Cx = eos.fugacity_coeffecient_VT(eos.x.values.tolist(), Px, Tx,'liquid')
    print("Cx", Cx)


    bubble = eos.bubble_pressure2()
    dummy, Zz, Cz = eos.fugacity_coeffecient_VT(eos.mole_fraction, bubble, T,'liquid')
    Vin = Zz * T / bubble - Cz
    Vout = eos.L * (Zx * Tx / Px - Cx)

    print(Vin, Vout)
    shrinkage = Vout / Vin

    print(shrinkage)
'''

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

    eos = VTPR(component_list, mole_fraction, MW_plus_fraction, SG_plus_fraction, P, T)

    Tx = (60 - 32) * 5 / 9 + 273.15
    Px = 1.0155
    eos.pt_flash(Px, Tx)
    print(eos.y)
    print(eos.x)
    print(eos.ZV)
    print(eos.ZL) 


    bubble = eos.bubble_pressure2()
    dummy, Zz = eos.fugacity_coeffecient(eos.mole_fraction, bubble, T,'liquid')
    Vin = Zz * T / bubble 
    Vout = eos.L * eos.ZL * Tx / Px

    print(Vin, Vout)
    shrinkage = Vout / Vin

    print(shrinkage)

    
    

    
    #print(eos.c)
    #eos._Twu_hybrid(T)
    #print(eos.MHV1(mole_fraction,P,T))

    #eos.MHV1(mole_fraction,P,T)


    #print(gibbs)

    '''
