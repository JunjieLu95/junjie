# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:21:01 2023

@author: jlu
"""

import os
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt
from mesopylib.extract.frq2 import frq2weyl
from mesopylib.utilities.load_save.loadXmlDat import loadXmlDat
from junjie.tools import get_lim, single_fit
from junjie.cpa.loading import load_dynamics, load_zero_num, load_Ta, load_ei_Heff
from junjie.cpa.rmt_dynamics import velocity_collect

def extract_zeros_rand_conf(fn_load):
    
    d=loadXmlDat(fn_load)
    frq_rand = d[0][0].getFrqAxis()
    S0=d[1]
    if S0.ndim==2:
        S0=S0[:,:,np.newaxis]
    
    zeros = np.array([])
    for i in range(len(fn_load)):
        S=S0[:,:,i]
        exp_S_rand = np.reshape(S, (4, 4, S.shape[1]), order='F')
        exp_S_rand = np.transpose(exp_S_rand,(2,0,1))
        inv_S_rand = np.linalg.inv(exp_S_rand)[:, 0, 1] 
        
        peaks_inds=scipy.signal.find_peaks(np.abs(inv_S_rand), height=2)[0]
        for j in range(len(peaks_inds)):
            range_fit = [peaks_inds[j]-20, peaks_inds[j]+20]
            frq_cut = frq_rand[range_fit[0]:range_fit[-1]]
            inv_S_cut = inv_S_rand[range_fit[0]:range_fit[-1]]
            
            try:
                zero_fit = single_fit(inv_S_cut, frq_cut)[1]
                if not np.isnan(zero_fit) and not np.abs(np.imag(zero_fit))>1:
                    zeros = np.append(zeros, zero_fit)
            except:
                pass
    return zeros
    

def fn_export(path0, Select=False, Perturbation=False, 
              Repeat=False, Random=False, All=False):
    """
    export the xmldat files
    for example: 
        path0 = 'D:/data/230110_Diff_CPA_4Ports_HumTemp/0.2-0.3/'
        fns = fn_export(path0, Repeat=True)
    """
    file_names = [Select, Perturbation, Repeat, Random, All]
    if np.sum(file_names)==1:
        if Select==True:
            path = path0 + 'SelectMeasure/'
        if Perturbation==True:
            path = path0 + 'Perturbation/'
        if Random==True:
            path = path0 + 'Randomconf/'
        if All==True:
            path = path0 + 'AllMeasure/'
        if Repeat==True:
            path = path0 + 'Repeat/'
            if os.path.exists(path):
                pass
            else:
                base_name = os.path.basename(path0[:-1])
                path = os.path.abspath(os.path.join(path0, '..')) + '/Repeat/'+ base_name + '/'
        if os.path.exists(path):
            files= os.listdir(path)
            files = sorted(files,  key=lambda x: os.path.getmtime(os.path.join(path, x)))
            fn = []
            for i in range(0, len(files)):
                if os.path.splitext(files[i])[1] == ".xmldat":
                    fn.append(path + files[i])
            return fn
        else:
            return False
    
def date_export(fns):
    """
    return the time difference (day) of the file
    """
    time = np.array([])
    for i in fns:
        time = np.append(time, os.path.getmtime(i))
    delta = (time - time[0])/3600/24
    return delta, time

def calc_evals_sorted(fn, port_use):
    """
    calculates the complex scattering matrix eigenvalues
    S : complex numpy array with S[channel1, channels2, frq]
    return evals : eigenvalues sorted as 
        evals[:,0] lowest eigenvalue as a function of frq
        ...
        evals[:,-1] highest eigenvalue as a function of frq
    """
    # create scattering matrix where first dimension is frequency 
    # to use linalg.ei for each frequency
    d=loadXmlDat(fn)
    frq = d[0][0].getFrqAxis()
    S=d[1]
    if port_use == 2:
        S = S[[0,2,8,10],:]
        exp_S = np.reshape(S, (2, 2, S.shape[1]), order='F')
    if port_use == 4:
        exp_S = np.reshape(S, (4, 4, S.shape[1]), order='F')
    if port_use == 22:
        exp_S = np.reshape(S, (2, 2, S.shape[1]), order='F')
    _tmp_S2 = np.transpose(exp_S,(2,0,1))
    evals = np.linalg.eig(_tmp_S2)[0]
    _ind_sort=np.argsort(np.abs(evals),axis=1)
    evals=np.take_along_axis(evals, _ind_sort, axis=1)
    return frq, evals

def S_export(fn, port_use):
    """
    reshape the S-matrix from the spectrum file
    """
    d=loadXmlDat(fn)
    S=d[1]
    if port_use == 2:
        S = S[[0,2,8,10],:]
        exp_S = np.reshape(S, (2, 2, S.shape[1]), order='F')
    if port_use == 4:
        exp_S = np.reshape(S, (4, 4, S.shape[1]), order='F')
    if port_use == 22:
        exp_S = np.reshape(S, (2, 2, S.shape[1]), order='F')
    exp_S = np.transpose(exp_S,(2,0,1))
    return exp_S

def Ta_plot(path, flag_std=True):
    """
    Calculate the transmission coefficient 
    Ta = 1-|<S>|^2

    """
    if os.path.exists(path):
        fns = fn_export(path, Random=True)
        d=loadXmlDat(fns)
        S_random_sum = d[1]
        sp = d[0][0].sParameters
        frq = d[0][0].getFrqAxis()
        Re_index = []
        for i in range(4):
            Re_index.append(i*5)
        
        plt.figure()
        for i in range(4):
            Ta = 1-np.abs(np.mean(S_random_sum[Re_index[i], :, :], axis=1))**2
            plt.plot(frq, Ta, label = f'{sp[Re_index[i]]}')
        plt.legend()
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('$T_a$')
        
        if flag_std:
            plt.figure()
            for i in range(4):
                plt.plot(frq, np.std(S_random_sum[Re_index[i], :, :], axis=1), label = f'{sp[Re_index[i]]}')
            plt.legend()
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('STD')
        # plt.savefig(path1 + r'Tc_randomconf.png', dpi=600, bbox_inches = 'tight')
    else:
        print('No such folder')

def Norm_complex_ei_3d_cavity(ei, V):
    """
    Normalize the complex eigenvalues by seperating the real and imag part 

    """
    ei_real = np.real(ei)
    ei_imag = np.imag(ei)
    ei_real_norm = frq2weyl(ei_real*1e9, V=V)
    ei_imag_norm = ei_imag/(constants.c**3/8/np.pi/(ei_real*1e9)**2/V/1e9)
    return  ei_real_norm+1j*ei_imag_norm

def N_3D_cavity(f, V):
    return 8*np.pi*V*(f*1e9)**3/(3*constants.c**3)

def num_of_Modes(f1, f2, V):

    f1=f1*1e9
    f2=f2*1e9
    N = 8*np.pi*V/(3*constants.c**3)*(f1**3-f2**3)
    return N

def mean_level_spacing(f, V):
    return constants.c**3/8/np.pi/f**2/V/1e9 # GHz

def interval_zeros_percent(zero_interval, w1, w2):
    n1 = len(get_lim(zero_interval, (w1, w2), 'imag'))
    n2 = len(zero_interval)
    per = np.around(n1/n2*100, 4)
    # print(per)
    return per

def P_num_zeros(path, nums, Im_range_lim, nbins, range_hist):
    """
    For numerical simulation, get the distribution of the imaginary part of the zeros,
    with the range (both real and imag) you define, to select the zeros,
    and calculate the histogram

    Parameters
    ----------
    path : string
        path of the data
    nums : list
        list of tuple, every tuple includes the info of the data
    Im_range_lim : tuple
        like (-30, 30)
    nbins : int
        bins of histogram
    range_hist: tuple
        the range of the histogram

    Returns
    -------
    dics : list
        list of tuple, new tuple includes the info and results of hist

    """
    dics = []
    for input_params in nums:
        N0 = input_params['N']
        kappa0 = input_params['kappa']
        num_zeros = load_zero_num(path + f'N{N0}/', input_params)
        
        Re_range_lim = (-N0*0.01, N0*0.01)
        num_zeros_ReLim = get_lim(num_zeros, Re_range_lim, which_complex='real')
        num_zeros_ReImLim = get_lim(num_zeros_ReLim, Im_range_lim, which_complex='imag')
        num_zeros_Im = np.imag(num_zeros_ReImLim)
        print('N zeros num:' + f'{len(num_zeros_Im)}')
        
        P_num_zeros = np.histogram(num_zeros_Im,  bins=nbins, density=True, range=range_hist)
        P_x_mean = (P_num_zeros[1][:-1] + P_num_zeros[1][1:])/2
        P_final = (P_num_zeros[0], P_x_mean)
        dic_data = {}
        dic_data['N'] = N0
        dic_data['kappa'] = kappa0
        dic_data['M'] = input_params['M']
        dic_data['EIm'] = input_params['EIm']
        dic_data['P_zero'] = P_final
        dics.append(dic_data)
    
    return dics

def P_exp_zeros(ex_zeros, V_rc, Im_range_lim, nbins, range_hist):
    """
    For experiment zeros data, first normlize the zeros,
    then select them and calculate the histogram

    Parameters
    ----------
    num_zeros : array
        the full zeros array
    Re_range_lim : tuple
        usually we choose (-1%N, 1%N).
    Im_range_lim : tuple
        like (-30, 30)
    nbins : int
        bins of histogram

    Returns
    -------
    P_num_zeros : tuple
    results of histogram

    """
    exp_zeros_Im = np.imag(Norm_complex_ei_3d_cavity(ex_zeros, V=V_rc))
    exp_zeros_ImLim = get_lim(exp_zeros_Im, Im_range_lim, which_complex='real')
    print('N zeros exp:' + f'{len(exp_zeros_ImLim)}')
    print(np.mean(exp_zeros_ImLim))
    P_exp_zeros = np.histogram(exp_zeros_ImLim,  bins=nbins, density=True, range=range_hist)
    P_x_mean = (P_exp_zeros[1][:-1] + P_exp_zeros[1][1:])/2
    return (P_exp_zeros[0], P_x_mean), exp_zeros_ImLim

def P_num_velocity(path, nums, Im_range_narrow=(0, 1), Im_range_super=(8.63, 33.63), nbins=51, range_hist=(-5, 5)):
    """
    For numerical simulation, get the distribution of the normlized velocity

    Parameters
    ----------
    path : string
        path of the data
    nums : list
        list of tuple, every tuple includes the info of the data
    Im_range_narrow : tuple
        define the Im range for narrow zeros
    Im_range_super : tuple
        define the Im range for super zeros
    nbins : int
        number of bins of the histogram
    range_hist: tuple
        the range of the histogram

    Returns
    -------
    dics : list
        list of tuple, new tuple includes the info and results of hist

    """
    dics = []
    for input_params in nums:
        N0 = input_params['N']
        kappa0 = input_params['kappa']
        width0 = input_params['width']
        data0 = load_dynamics(path + f'{width0}/N_{N0}_{kappa0}/', input_params)
        print(f'number of realization: {np.shape(data0)[1]}')
        
        if width0 == 'narrow':
            cutRange_Im = Im_range_narrow
        elif width0 == 'super':
            cutRange_Im = Im_range_super
        cutRange_Re = [-0.01*N0, 0.01*N0]
        zero0 = np.conj(data0)
        v = velocity_collect(zero0, rescale=False, Erange=cutRange_Re, cutRange=cutRange_Im, flag_exp=False)
        
        v_type = 0   # 0, 1, 2 means real, imag, abs part of v
        # std_v = np.std(v[v_type])
        # v_norm = v[v_type]/std_v
        v_re_N = v[v_type]/np.sqrt(N0)
        std_v = np.std(v_re_N)
        v_norm = v_re_N/std_v

        
        P_v = np.histogram(v_norm, bins=nbins, density=True, range=range_hist)
        P_v_mean = (P_v[1][:-1] + P_v[1][1:])/2
        P_v_final = (P_v[0], P_v_mean)
        
        dic_data = {}
        dic_data['N'] = N0
        dic_data['kappa'] = kappa0
        dic_data['width'] = width0
        dic_data['std_v'] = std_v
        dic_data['P_v'] = P_v_final
        dics.append(dic_data)
    return dics

def P_exp_velocity(exp_zeros, V_rc, nbins, range_hist):
    """
    For experiment, get the distribution of the normlized velocity

    Parameters
    ----------
    exp_zeros : array
        experimental data of extracted zeros
    V_rc : float
        the size of the chamber
    nbins : int
        number of bins of the histogram
    range_hist: tuple
        the range of the histogram

    Returns
    -------
    P_exp_finaly : tuple
        hist of distribution

    """
    zeros_weyl = Norm_complex_ei_3d_cavity(exp_zeros, V=V_rc)
    zeros_weyl = np.transpose(zeros_weyl)
    
    v_exp = velocity_collect(zeros_weyl, flag_exp=True)
    v_type = 0   # 0, 1, 2 means real, imag, abs part of v
    std_exp = np.std(v_exp[v_type])
    v_exp_norm = v_exp[v_type]/std_exp
    P_exp = np.histogram(v_exp_norm, bins=nbins, density=True, range=range_hist)
    P_exp_mean = (P_exp[1][:-1] + P_exp[1][1:])/2
    return (P_exp[0], P_exp_mean), std_exp

def cal_num_Ta(path, nums):
    dics = []
    for input_params in nums:
        data0 = load_Ta(path, input_params)
        ta = np.mean(data0)
        
        dic_data = {}
        dic_data['N'] = input_params['N']
        dic_data['kappa'] = input_params['kappa']
        dic_data['M'] = input_params['M']
        dic_data['EIm'] = input_params['EIm']
        dic_data['Ta'] = ta
        dics.append(dic_data)
    return dics

def P_num_ei_Heff(path, nums, nbins, nbins_Re=251, w1=-1.35, w2=12):
    """
    For numerical simulation, get the distribution of the imaginary part of the poles,
    with the range (both real and imag) you define, to select the zeros,
    and calculate the histogram

    Parameters
    ----------
    path : string
        path of the data
    nums : list
        list of tuple, every tuple includes the info of the data
    nbins : int
        bins of histogram

    Returns
    -------
    dics : list
        list of tuple, new tuple includes the info and results of hist

    """
    dics = []
    for input_params in nums:
        N0 = input_params['N']
        kappa0 = input_params['kappa']
        shift = -input_params["EIm"]
        num_ei = load_ei_Heff(path, input_params)
        num_ei = np.conj(num_ei)+shift*1j
        
        Re_range_lim = (-N0*0.01, N0*0.01)
        num_ei_ReLim = get_lim(num_ei, Re_range_lim, which_complex='real')
        num_ei_Im = np.imag(num_ei_ReLim)
        print('N zeros num:' + f'{len(num_ei_Im)}')
        if len(num_ei_Im)>0:
            per = interval_zeros_percent(num_ei_ReLim, w1, w2)
        else:
            per = 0
            
        range_Im = input_params.get('range_Im', None)

        P_num_ei_Im = np.histogram(num_ei_Im, bins=nbins, density=True, range=range_Im)
        P_x_mean_Im = (P_num_ei_Im[1][:-1] + P_num_ei_Im[1][1:])/2
        P_final_Im = (P_num_ei_Im[0], P_x_mean_Im)
        
        par_scale = input_params.get('Re_scale', 1)
        num_ei_Re = np.real(num_ei)*par_scale
        P_num_ei_Re = np.histogram(num_ei_Re, bins=nbins_Re, density=True)
        P_x_mean_Re = (P_num_ei_Re[1][:-1] + P_num_ei_Re[1][1:])/2
        P_final_Re = (P_num_ei_Re[0], P_x_mean_Re)
        
        dic_data = {}
        dic_data['N'] = N0
        dic_data['kappa'] = kappa0
        dic_data['M'] = input_params['M']
        dic_data['P_zero_Im'] = P_final_Im
        dic_data['P_zero_Re'] = P_final_Re
        dic_data['percent'] = per
        dics.append(dic_data)
    return dics

def get_prob_complex_plane(zero_c, n=101):
    '''
    Input: the zeros and the size of the mesh
    Output: the probability of the zeros in the complex plane
    '''
    xedges = np.linspace(np.min(np.real(zero_c)), np.max(np.real(zero_c)), n)
    yedges = np.linspace(np.min(np.imag(zero_c)), np.max(np.imag(zero_c)), n)
    H, xedges, yedges = np.histogram2d(np.real(zero_c), np.imag(zero_c), bins=(xedges, yedges))
    H = H.T
    H[H==0] = np.nan
    H = H/np.nansum(H)
    X, Y = np.meshgrid(xedges, yedges)
    return (X,Y,H)

def get_prob_complex_plane_logspace(zero_c, n=101):
    '''
    Input: the zeros and the size of the mesh
    Output: the probability of the zeros in the complex plane
    '''
    xedges = np.linspace(np.min(np.real(zero_c)), np.max(np.real(zero_c)), n)
    yedges = -np.logspace(np.log(-np.min(np.imag(zero_c))), np.log(-np.max(np.imag(zero_c))), n)
    H, xedges, yedges = np.histogram2d(np.real(zero_c), np.imag(zero_c), bins=(xedges, yedges))
    H = H.T
    H[H==0] = np.nan
    H = H/np.nansum(H)
    X, Y = np.meshgrid(xedges, yedges)
    return (X,Y,H)