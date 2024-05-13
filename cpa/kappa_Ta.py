# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:17:01 2024

@author: jlu
"""
import numpy as np
from sympy import symbols, nsolve

def Ta_to_kappa(Ta, flag_overcoupled = True):
    """
    Transform the Ta to kappa, depends on the flag, the return is different
    (without global loss)

    """
    a=np.array(Ta)
    b=2*np.array(Ta)-4
    c=np.array(Ta)
    kappa_all=[(-b+np.sqrt(b**2-4*a*c))/2/a, (-b-np.sqrt(b**2-4*a*c))/2/a]
    if flag_overcoupled:
        return kappa_all[0]
    else:
        return kappa_all[1]
    
def kappa_to_Ta(kappa):
    """
    Transform the kappa to Ta (without global loss)

    """
    kappa=np.array(kappa)
    return 4*kappa/((1+kappa)**2)

def g(N, M, Gamma):
    m = M/N
    lambda0 = N/np.pi
    return -Gamma/4/lambda0+np.sqrt((1-m)+Gamma**2/16/(lambda0**2))

def cal_kappa_loss(N, M, Gamma, Ta, flag_over=True):
    """
    When the system with global loss, the kappa and Ta do not follow standard relation

    """
    
    kappa_eff = symbols('kappa_eff')
    
    g0 = g(N, M, Gamma)
    
    saa = (1 - g0 * kappa_eff) / (1 + g0 * kappa_eff)
    Ta_equation = 1 - saa**2 - Ta
    if Gamma == 37.26:
        if flag_over:
            if N <= 15:  
                initial_guess = 10  
            elif N <= 100:  
                initial_guess = 4  
            else:
                initial_guess = 2
        else:
            initial_guess = 0.5
    elif Gamma == 20:
        if flag_over:
            if N <= 15:  
                initial_guess = 10  
            elif N <= 100:  
                initial_guess = 4  
            else:
                initial_guess = 2
        else:
            initial_guess = 0.5
            
    elif Gamma == 0:
        if flag_over:
            initial_guess = 1.5
        else:
            initial_guess = 0.7
        
    kappa_eff_value = nsolve(Ta_equation, kappa_eff, initial_guess) 
    return float(kappa_eff_value)

def cal_Ta_loss(N, M, Gamma, kappa):
    """
    When the system with global loss, the kappa and Ta do not follow standard relation

    """
    g0 = g(N, M, Gamma)
    saa = (1 - g0 * kappa) / (1 + g0 * kappa)
    Ta = 1 - saa**2 
    return Ta

    