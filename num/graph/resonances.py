# -*- coding: utf-8 -*-
"""
Extract the resonances of scattering matrix
"""
import numpy as np
import matplotlib.pyplot as plt
from junjie.tools import mkdir
from skimage.feature import peak_local_max

def k_window(kr_cal):
    """
    Give the k-window for the computation

    Parameters
    ----------
    kr_cal : tuple
        (k_start, k_end, k_step)
    Returns
    -------
    k_window : list
        The window for computing

    """
    (k_start, k_end, k_step) = kr_cal
    
    k_window = []
    for i in range(int((k_end-k_start)/k_step)+1):
        k_window.append(k_start+k_step*i)
    k_window.append(k_end)
    return k_window

def extract_func_peak(kr_cal, ki_estimate, graph_info=None, kr_tolrance=0.2, 
                      ki_tolrance=0.05, cal_func=None, extract_type='poles', 
                      n_discrete=500, flag_plot=False):
    """
    Use Func: peak_local_max to find the local max of |S| in complex plane

    Parameters
    ----------
    kr_cal : tuple
       The real part range for calculate the resonances (k_start, k_end, k_step)
    ki_estimate : tuple
        The real part range for calculate the resonances  (ki_start, ki_end), small than 0
    graph_info : list, optional
        including length, or phase of the graph
        Must be consistent with the Func cal_func!!
        Sometimes graph info is no need, since it already includes in the cal_func
    kr_tolrance : float
        tolrance for kr range, to compute the resonances near the boundary
    ki_tolrance : float
        tolrance for ki range, to compute the resonances near the boundary
    cal_func : function
        A function return the scattering matrix for computing
    extract_type : str, optional
        'poles' or 'zeros', now it is for completeness
    n_discrete : int, optional
        the grid of the complex k

    Returns
    -------
    resonances : list
        length of the list: length of phis
        the element of the list is the array, 
        which is the resonances in different positions

    """
    k_sep = k_window(kr_cal)
    
    resonances = np.array([])
    for i in range(len(k_sep)-1):
        kr = np.linspace(k_sep[i]+0.001-kr_tolrance, k_sep[i+1]+kr_tolrance, n_discrete)
        ki = np.linspace(ki_estimate[0]-ki_tolrance, ki_estimate[1]+ki_tolrance, n_discrete)
        krm, kim = np.meshgrid(kr, ki)
        nr = len(kr)
        ni = len(ki)
        kri = np.reshape(krm + kim*1j, (nr*ni))
        
        if graph_info:
            S = cal_func(kri, *graph_info)
        else:
            S = cal_func(kri)
        
        S = np.reshape(S, (ni, nr, 1, 1))

        S = np.abs(S[:,:,0,0])
        
        local_mp = peak_local_max(S)
        local_mpx = local_mp[:,0]
        local_mpy = local_mp[:,1]
        
        res = ki[local_mpx]*1j+kr[local_mpy]
        lim = (res.real>k_sep[i]) & (res.real<k_sep[i+1]) & (res.imag>ki_estimate[0]) & (res.imag<ki_estimate[1])
        resonances=np.append(resonances, np.sort(res[lim]))
        
        if flag_plot:
            plt.figure()
            plt.pcolormesh(np.transpose(krm), np.transpose(kim), np.transpose(S), shading='auto', cmap=plt.cm.gray)
            
    return resonances

def cal_poles_pars(kr_cal, ki_estimate, graph_infos=None, pars_ind=None,
                  kr_tolrance=0.2, ki_tolrance=0.05, cal_func=None, 
                  extract_type='poles', n_discrete=500, **kwargs):
    """
    for parametrically varying the graphs and compute the resonances.
    """
    r = []
    n = np.shape(graph_infos)[0]
    m = np.shape(graph_infos)[1]
    for i in range(n):
        r1 = []
        # print(i)
        for j in range(m):
            graph_info = graph_infos[i,j]
            r2 = extract_func_peak(kr_cal, ki_estimate, graph_info, kr_tolrance, 
                                   ki_tolrance, cal_func, extract_type, n_discrete)
            r1.append(r2)
        r.append(r1)
    return r

def plot_poles(resonances, xlim, ylim, save_path=False, marker_phase=[-1,-1], color='b'):
    """
    Plot the poles in the complex plane

    """
    mkdir(save_path)
    for n, i in enumerate(resonances):
        # plt.figure('EP')
        # plt.clf()
        # plt.figure(figsize=(10,4))
        # plt.clf()
        for m, j in enumerate(i):
            if m==marker_phase[0]:
                plt.plot(np.real(j), -np.imag(j), ls=' ', marker='^', color='r')
            elif m==marker_phase[1]:
                plt.plot(np.real(j), -np.imag(j), ls=' ', marker='s', color='g')
            else:
                plt.plot(np.real(j), -np.imag(j), ls=' ', marker='.', color=color)
                    
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(r'$\mathrm{Re}(k)$')
        plt.ylabel(r'$-\mathrm{Im}(k)$')
        if save_path:
            plt.savefig(save_path + f'{n}.png', dpi=300)