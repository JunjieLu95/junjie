# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:21:01 2023

@author: jlu
"""

import os
import numpy as np
import scipy
from scipy import constants
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mesopylib.extract.frq2 import frq2weyl
from mesopylib.extract.cLorentzFit import cFitStep,cLorentz
from mesopylib.utilities.load_save.loadXmlDat import loadXmlDat
from junjie.tools import get_lim, single_fit
from junjie.cpa.data_processing import 

        
def Temp_Hum_load(path0, name, path_code, flag_plot=True, flag_fit_temp=False, flag_oscillation=True):
    """
    Load the file of temperature and humdity, there are two kinds of data,
    
    one is the old one, we only use one sensor (DHT22) to measure,
    the other is new setup, we use six sensors (SHT35) to measure, 
    and these six sensors are loacted in different positions of outside of the chamber
    
    the size of the array should be like (30000, 9, 6, 2)
    30000 indicates the number of data points
    9 indicates the number of optimizations you are looking at,
    6 indicates the number of the sensor,
    2 indicates the (temperature and humdity)
    
    Returns
    -------
    n_points: due to the stop of measurement, the number of spectrums may be different,
    so this index is to cut and make sure every set have same number
    
    """
    fns = fn_export(path0 + name + '/', Select=False, Repeat=True)
    n_points = len(fns)-1
    days, times = date_export(fns)
    days = days[:n_points]
    
    path = path0 + 'Repeat/'
    if os.path.exists(path):
        HumTemp=np.load(path0 + 'Repeat/HumTemp.npy', allow_pickle=True, fix_imports=True)
    else:
        HumTemp=np.load(path0 + 'HumTemp.npy', allow_pickle=True, fix_imports=True)
        
    Hum = np.mean(HumTemp[:n_points,:,:,0], axis=2)
    Temp = np.mean(HumTemp[:n_points,:,:,1], axis=2)
    Temp_plot = Temp[:,0]
    Hum_plot = Hum[:,0]
    
    if flag_fit_temp:
        if flag_oscillation:                           # 230110 data
            peaks, _ = find_peaks(Temp_plot, distance=150)
            peaks[0]=0
            peaks=np.append(peaks, n_points)
            peaks=peaks[::2]
            peaks=np.append(peaks, n_points)
            Temp_fit =  np.array([])
            for i in range(len(peaks)-1):
                x = np.arange(len(Temp_plot[peaks[i]:peaks[i+1]]))
                z = np.polyfit(x, Temp_plot[peaks[i]:peaks[i+1]], 11) 
                Temp_fit = np.append(Temp_fit, np.polyval(z,x))
        else:
            x = np.arange(len(Temp_plot))
            z=np.polyfit(x, Temp_plot, 5) 
            Temp_fit=np.polyval(z,x)
    
    if flag_plot:
        plt.figure('temp', figsize = (15,6))
        plt.plot(days, Temp_plot)
        if flag_fit_temp:
            plt.plot(days, Temp_fit)
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Temperature', fontsize=12)
        plt.savefig(path_code + 'pic_230505/temperature_change.png', dpi=600)
        
        plt.figure(figsize = (15,6))
        plt.plot(days, Hum_plot)
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Humidity', fontsize=12)
        plt.savefig(path_code + 'pic_230505/humidity_change.png', dpi=600)
    
    return Temp, Hum, n_points
            
def spectrum_compare(path0, name, element_ind, flag_s_inv, flag_repeat=True, **kwargs):
    """
    Plot of compare two spectrums, if flag_repeat=True, we compare the spectrums
    from repeat measurement, otherwise, we compare it from optimize(select) measurement

    """
    if flag_repeat:
        fns = fn_export(path0 + name + '/', Select=False, Repeat=True)
    else:
        fns = fn_export(path0 + name + '/', Select=True, Repeat=False)
    if element_ind==None:
        element_ind=[0,1]
    if isinstance(fns, list):
        frq, _ = calc_evals_sorted(fns[-1], 4)
        S_plot = np.zeros((2, len(frq)), complex)
        for j in range(2):
            fn_ind = [0, -1]
            exp_S = S_export(fns[fn_ind[j]], 4)
            if flag_s_inv:
                S_plot[j] = np.linalg.inv(exp_S)[:, element_ind[0], element_ind[1]]
            else:
                S_plot[j] = exp_S[:, element_ind[0], element_ind[1]]
            
        plt.figure()
        plt.plot(frq, np.abs(S_plot[0])**2, label='first')  
        plt.plot(frq, np.abs(S_plot[-1])**2, label='last')      
        plt.xlabel('Frequency (GHz)')
        if flag_s_inv:
            plt.ylabel(r'$|(S^{-1})|^2$')
            plt.title(f'inv S-matrix {name}')
        else:
            plt.ylabel(r'$|S|^2$')
            plt.title(f'S-matrix {name}')
        plt.legend()

def temp_zeros_plot(Temp, zeros, path_code, name, flag_near_deg, flag_s_inv, flag_linear_fit=True, fit_range=None, flag_zeros_norm=False, **kwargs):
    
    if flag_s_inv:
        temp_plot = Temp
        zeros_plot = zeros
        if flag_zeros_norm:
            zeros_plot = Norm_complex_ei_3d_cavity(zeros_plot)
        
        plt.figure()
        if zeros_plot.ndim==1:
            plt.plot(temp_plot, np.real(zeros_plot))
            plt.scatter(temp_plot[0], np.real(zeros_plot)[0], s=50, marker='o', color='r', zorder=100)
            plt.scatter(temp_plot[-1], np.real(zeros_plot)[-1], s=50, marker='*', color='orange', zorder=100)
        elif zeros_plot.ndim==2:
            for i in range(2):
                plt.plot(temp_plot, np.real(zeros_plot[i]))
                plt.scatter(temp_plot[0], np.real(zeros_plot[i])[0], s=50, marker='o', color='r', zorder=100)
                plt.scatter(temp_plot[-1], np.real(zeros_plot[i])[-1], s=50, marker='*', color='orange', zorder=100)
        
        plt.title(f'Temp-zeros {name}')
        plt.xlabel('Temperature')
        plt.ylabel(r'$\mathrm{Re}(E)$')
        if fit_range==None:
            fit_range = [0, -1]
        if flag_linear_fit:
            if zeros_plot.ndim==1:
                x=temp_plot[~np.isnan(zeros_plot)][fit_range[0]:fit_range[1]] 
                y=np.real(zeros_plot[~np.isnan(zeros_plot)])[fit_range[0]:fit_range[1]] 
                z=np.polyfit(x, y, 1)
                fit=np.polyval(z,x)
                print(z)
                plt.plot(x, fit)
            elif zeros_plot.ndim==2:
                for i in range(2):
                    x=temp_plot[~np.isnan(zeros_plot[i])][fit_range[0]:fit_range[1]] 
                    y=np.real(zeros_plot[i][~np.isnan(zeros_plot[i])])[fit_range[0]:fit_range[1]] 
                    z=np.polyfit(x, y, 1)
                    fit=np.polyval(z,x)
                    print(z)
                    plt.plot(x, fit)
        plt.savefig(path_code + f'pic_230505/temp_zeros_{name}.png', dpi=600)

def temp_frq_plot(Temp, zeros, path_code, name, flag_near_deg, flag_s_inv, flag_linear_fit=True, fit_range=None, flag_zeros_norm=False, **kwargs):
    
    temp_plot = Temp
    zeros_plot = zeros
    if flag_zeros_norm:
        zeros_plot = Norm_complex_ei_3d_cavity(zeros_plot)
    
    plt.figure()
    if zeros_plot.ndim==1:
        plt.plot(temp_plot, np.real(zeros_plot))
        plt.scatter(temp_plot[0], np.real(zeros_plot)[0], s=50, marker='o', color='r', zorder=100)
        plt.scatter(temp_plot[-1], np.real(zeros_plot)[-1], s=50, marker='*', color='orange', zorder=100)
    elif zeros_plot.ndim==2:
        for i in range(2):
            plt.plot(temp_plot, np.real(zeros_plot[i]))
            plt.scatter(temp_plot[0], np.real(zeros_plot[i])[0], s=50, marker='o', color='r', zorder=100)
            plt.scatter(temp_plot[-1], np.real(zeros_plot[i])[-1], s=50, marker='*', color='orange', zorder=100)
    
    plt.title(f'Temp-frq {name}')
    plt.xlabel('Temperature')
    plt.ylabel(r'$\mathrm{Re}(E)$')
    if fit_range==None:
        fit_range = [0, -1]
    if flag_linear_fit:
        if zeros_plot.ndim==1:
            x=temp_plot[~np.isnan(zeros_plot)][fit_range[0]:fit_range[1]] 
            y=np.real(zeros_plot[~np.isnan(zeros_plot)])[fit_range[0]:fit_range[1]] 
            z=np.polyfit(x, y, 1)
            fit=np.polyval(z,x)
            print(z)
            plt.plot(x, fit)
        elif zeros_plot.ndim==2:
            for i in range(2):
                x=temp_plot[~np.isnan(zeros_plot[i])][fit_range[0]:fit_range[1]] 
                y=np.real(zeros_plot[i][~np.isnan(zeros_plot[i])])[fit_range[0]:fit_range[1]] 
                z=np.polyfit(x, y, 1)
                fit=np.polyval(z,x)
                print(z)
                plt.plot(x, fit)
    plt.savefig(path_code + f'pic_230505/temp_frq_{name}.png', dpi=600)
        
def extract_min_pos(n_points, path0, name, element_ind, flag_near_deg, flag_s_inv, **kwargs):
    
    fns = fn_export(path0 + name + '/', Select=False, Repeat=True)
    frq, _ = calc_evals_sorted(fns[-1], 4)
    if element_ind==None:
        element_ind=[0,1]
    if flag_s_inv:
        if flag_near_deg:
            frqs = np.zeros((2, n_points))
            for i in range(n_points):
                _, evals = calc_evals_sorted(fns[i], 4)
                # frqs[i] = frq[np.argmin(np.abs(evals[:,0]))]
                
                signal_test1 = np.abs(evals[:,0])
                z=scipy.signal.find_peaks(1/signal_test1, height=0.1, distance = 50)[0]
                cut_spectrum_ind = z[(z>400) & (z<len(frq)-400)]
                
                center_ind = np.int(len(frq)/2)
                for j in range(len(cut_spectrum_ind)):
                    if cut_spectrum_ind[j]<center_ind:
                        frqs[0,i] = frq[cut_spectrum_ind[j]]
                    else:
                        frqs[1,i] = frq[cut_spectrum_ind[j]]
                
        else:
            frqs = np.zeros(n_points)
            for i in range(n_points):
                _, evals = calc_evals_sorted(fns[i], 4)
                frqs[i] = frq[np.argmin(np.abs(evals[:,0]))]
    else:
        frqs = np.zeros(n_points)
        for i in range(n_points):
            exp_S = S_export(fns[i], 4)[:, element_ind[0], element_ind[1]]            
            frqs[i] = frq[np.argmax(np.abs(exp_S))]
    
    return frqs

def extract_zeros_temperature(n_points, path0, name, range_fit, element_ind, flag_near_deg, flag_s_inv, **kwargs):
    """
    Try to extract zeros from the inv(s), and we deal it different when the zeros are near degenerated 

    """
    
    fns = fn_export(path0 + name + '/', Select=False, Repeat=True)
    frq, _ = calc_evals_sorted(fns[-1], 4)
    if element_ind==None:
        element_ind=[0,1]
    if flag_s_inv:
        if flag_near_deg:
            zeros = np.zeros((2, n_points), complex)+np.nan+np.nan*1j
            for i in range(n_points):
                print(i)
                exp_S = S_export(fns[i], 4)
                inv_S = np.linalg.inv(exp_S)[:, element_ind[0], element_ind[1]]
                
                signal_test1 = np.abs(inv_S)
                z=scipy.signal.find_peaks(signal_test1, height=30, distance = 50)[0]
                cut_spectrum_ind = z[(z>400) & (z<len(frq)-400)]
    
                for j in range(len(cut_spectrum_ind)):
                    range_lim = [cut_spectrum_ind[j]-100, cut_spectrum_ind[j]+100]
                    frq_fit = frq[range_lim[0]:range_lim[-1]]
                    inv_S_fit = inv_S[range_lim[0]:range_lim[-1]]
                    try:
                        zero_fit=single_fit(inv_S_fit, frq_fit)[1]
                        if not np.isnan(zero_fit) and not np.abs(np.imag(zero_fit))>100:
                            if zero_fit.real<(frq[z[0]]+frq[z[1]])/2:
                                zeros[0,i] = zero_fit
                            else:
                                zeros[1,i] = zero_fit
                    except:
                        zeros[0,i] = np.nan+np.nan*1j
                        zeros[1,i] = np.nan+np.nan*1j
        else:
            zeros = np.zeros(n_points, complex)+np.nan+np.nan*1j
            if range_fit==None:
                center_ind = np.int(len(frq)/2)
                range_fit = [center_ind-700, center_ind+700]
            for j in range(n_points):
                print(j)
                exp_S = S_export(fns[j], 4)
                inv_S = np.linalg.inv(exp_S)[:, element_ind[0], element_ind[1]]
                frq_cut = frq[range_fit[0]:range_fit[-1]]
                inv_S_cut = inv_S[range_fit[0]:range_fit[-1]]
                try:
                    zero_fit = single_fit(inv_S_cut, frq_cut)[1]
                    if not np.isnan(zero_fit) and not np.abs(np.imag(zero_fit))>100:
                        zeros[j] = zero_fit
                except:
                    zeros[j] = np.nan+np.nan*1j
    else:
        zeros = np.zeros(n_points)+np.nan
        
    return zeros