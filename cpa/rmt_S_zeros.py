# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:45:11 2021

@author: Junjie
"""

import numpy as np
import scipy.signal
import time
from datetime import timedelta
from junjie.tools import single_fit
from junjie.cpa.saving import path_base, fn_save_Ta
import mesopylib.num.rmt.open_systems.S_matrix.construction as constr

def multi_fit(E, S):
    """
    Use multiple single fits to extract zeros in a large frq window (for numerical simulation)

    Parameters
    ----------
    E : array
        Energy
    S : array
        s-matrix

    Returns
    -------
    zeros : list
        extracted zeros

    """
    zeros=[]
    inv_S_rand = np.linalg.inv(S)[:, 0, 1] 
    # plt.plot(np.real(E), np.abs(inv_S_rand))
    peaks_inds=scipy.signal.find_peaks(np.abs(inv_S_rand), height=2)[0]   #prominence=0.1
    for j in range(len(peaks_inds)):
        range_lim = [peaks_inds[j]-600, peaks_inds[j]+600]
        inv_S_peak = inv_S_rand[range_lim[0]:range_lim[-1]]
        frq_peak = E.real[range_lim[0]:range_lim[-1]]
        try:
            zeros.append(single_fit(inv_S_peak, frq_peak)[1])
            # plt.plot(frq_peak, np.abs(cLorentz(frq_peak, single_fit(inv_S_peak, frq_peak))))
        except:
            pass
    return zeros

def main_get_zeros(M, N, kappa, E, flag_noise=False):
    """
    get the s-matrix and extract zeros

    Parameters
    ----------
    M : int
        number of channels
    N : int
        size of Heff
    kappa : float
        coupling strength.
    E : array
        input energy
    flag_noise : bool, optional
        add noise to the s-matrix or not. The default is False.

    Returns
    -------
    z1, z2
        the extracted zeros from s-matrix without and with noise.

    """
    sys1 = constr.create_S(M, N, [kappa], seed_H=None, seed_V=None, Provider=constr.LargeNSMatrix)
    S=sys1.S(E)
    z1 = multi_fit(E, S)
    z2 = []
    if flag_noise:
        S_n = S.copy()
        for i in range(2):
            for j in range(2):
                S_n[:,i,j] += 2e-6*(np.random.uniform(low=-1, high=1, size=(len(E),))
                                  + np.random.uniform(low=-1, high=1, size=(len(E),)) * 1j)
        z2 = multi_fit(E, S_n)
    return z1, z2

def cal_S(n_real=1000, Gamma_abs = [0,0.1], N=1000, kappa = [1, 0.005], M = [1,2], E=0):
    """
    n_real=100 : Number of Realizations
    Gamma_abs = 0.1 : uniform absorption
    N :_ Size of H
    kappa = [1, 0.005] : Antenna couplings
    M = [1,2] : Number of Antennas with coupling kappa[0],kappa[1],...

    """

    SE0=[]
    for i in np.arange(n_real):
        sys1 = constr.create_S(M, N, kappa, seed_H=None, seed_V=None, Provider=constr.LargeNSMatrix)
        iS=[]
        for iGamma_abs in Gamma_abs:
            iS.append(sys1.S(E + 1j * iGamma_abs)) 
        SE0.append(iS)
    SE0=np.array(SE0)
    return SE0

def cal_Ta(N, M, Gamma, kappa, n_real=20000, nn=101, path="", flag_save=True):
    """
    Compute tas for varying parameters.

    """
    if not isinstance(N, np.ndarray):
        N = np.array([N])
    if not isinstance(M, np.ndarray):
        M = np.array([M])
    if not isinstance(Gamma, np.ndarray):
        Gamma = np.array([Gamma])
    if not isinstance(kappa, np.ndarray):
        kappa = np.array([kappa])
    
    for n_value in N:
        for m_value in M:
            for gamma_value in Gamma:
                for kappa_value in kappa:
                    E = np.array([gamma_value * 1j / 2])
                    tas = []
                    begintime = time.time()

                    for ii in range(nn):
                        S0 = cal_S(n_real=n_real, Gamma_abs=np.array([0]), N=n_value, kappa=[kappa_value], M=np.array([m_value]), E=E)
                        ta = np.mean(1 - np.abs(np.mean(S0[:, 0, 0, 0, 0]))**2)
                        tas.append(ta)
                    avg_tas = np.mean(tas)
                    print(f"Average T_a for N={n_value}, M={m_value}, Gamma={gamma_value}, kappa={kappa_value}: {avg_tas}")
                    
                    final_time = time.time()
                    total_time = final_time - begintime
                    formatted_time = str(timedelta(seconds=total_time))
                    print(f"Running time: {formatted_time}")
                    
                    if flag_save:
                        fn = fn_save_Ta(path, n_value, m_value, n_real * nn, kappa_value, E)
                        np.save(fn, tas, allow_pickle=True, fix_imports=True)
                        
