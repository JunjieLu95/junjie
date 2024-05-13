# -*- coding: utf-8 -*-
"""
This program is used for calculating the S-matrix of the open GOE Neumann graph.
For this purpose, first we need get the h maxtrix, then we can get the scattering matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
from junjie.tools import mkdir
from skimage.feature import peak_local_max

def k_window(kr_cal):
    [k_start, k_end, k_step] = kr_cal
    
    k_window = []
    for i in range(np.int((k_end-k_start)/k_step)+1):
        k_window.append(k_start+k_step*i)
    k_window.append(k_end)
    return k_window

def extract_func_peak(kr_cal, ki_estimate, graph_info=None, kr_tolrance=0.2, 
                      ki_tolrance=0.05, cal_func=None, extract_type='poles', 
                      n_discrete=500):
    
    k_sep = k_window(kr_cal)
    
    resonances = np.array([])
    for i in range(len(k_sep)-1):
        kr = np.linspace(k_sep[i]-kr_tolrance, k_sep[i+1]+kr_tolrance, n_discrete)
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
        
        # plt.figure()
        # plt.pcolormesh(np.transpose(krm), np.transpose(kim), np.transpose(S), shading='auto', cmap=plt.cm.gray, vmin=None, vmax=10)
        
    return resonances

def cal_poles_pars(kr_cal, ki_estimate, graph_infos=None, pars_ind=None,
                  kr_tolrance=0.2, ki_tolrance=0.05, cal_func=None, 
                  extract_type='poles', n_discrete=500, **kwargs):
    
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
    mkdir(save_path)
    for n, i in enumerate(resonances):
        plt.figure('EP')
        plt.clf()
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