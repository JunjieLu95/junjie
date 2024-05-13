# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:11:58 2023

@author: jlu
"""

import numpy as np

def new_spectrum_phi(kr, kim, phi0, S):
    """
    Update: in 2023.03.28 arg_max_neg-1 is wrong, fixed to arg_max_neg+1
    
    In a transformed spectrum ensembles (length to pahse), 
    before we extract resonances, we need first extract new spectrum with constant phase,
    this function is used for extract the wavenumber and do the fitting and get the S-matrix

    Parameters
    ----------
    kr : array
        array of wavenumber
    kim : array
        meshgrid of kr and ki (phase)
    phi0 : float
        phase value
    S : array
        same dims as krm and kim

    """
    if kim[1,0]-kim[0,0]>0:
        ind0  = np.argmin(np.abs(kim[-1]-phi0))+4    #find the start frq index of the new spctrum, "+4" is used for avoid bug
        ind1  = np.argmin(np.abs(kim[0]-phi0))       #find the end frq index of the new spctrum
        select = 1
    else:
        ind1  = np.argmin(np.abs(kim[-1]-phi0))-4    #find the start frq index of the new spctrum, "+4" is used for avoid bug
        ind0  = np.argmin(np.abs(kim[0]-phi0))       #find the end frq index of the new spctrum
        select = -1
        
    kr_choose = kr[ind0:ind1]
    s_choose = np.zeros(ind1-ind0, dtype=complex)
    for i in range(ind0, ind1):
        dphi_find_choose = kim[:, i]
        diff = dphi_find_choose - phi0
        arg_max_neg = np.where(np.max(diff[diff<0])==diff)[0][0]   #find the phase index of the point close to phi0, the value of the point is maximum of the negative value
        xp = [kim[arg_max_neg,i], kim[arg_max_neg+select,i]]              #find the maximum of the negative value and minimum of the positive value
        fp = [S[arg_max_neg,i], S[arg_max_neg+select,i]]
        s_choose[i-ind0] = np.interp(phi0, xp, fp)   #do the fitting
    return kr_choose, s_choose