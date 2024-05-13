# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 13:39:10 2023

@author: jlu
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from mesopylib.extract.caldata import caldata
import mesopylib.extract.harmonic_inversion as hi
from mesopylib.extract.cLorentzFit import cFitting,cLorentz

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
        diff0 = kim[-1]-phi0
        diff1 = phi0-kim[0]
        select = 1
    else:
        diff0 = kim[0]-phi0
        diff1 = phi0-kim[-1]
        select = -1
        
    try: 
        ind_min  = np.where(np.min(diff0[diff0>0])==diff0)[0][0]
        ind_max  = np.where(np.min(diff1[diff1>0])==diff1)[0][0]+1
    except:
        return []
    kr_choose = kr[ind_min:ind_max]
    s_choose = np.zeros(ind_max-ind_min, dtype=complex)
    for i in range(ind_min, ind_max):
        dphi_find_choose = kim[:, i]
        diff = dphi_find_choose - phi0
        arg_max_neg = np.where(np.max(diff[diff<0])==diff)[0][0]   #find the phase index of the point close to phi0, the value of the point is maximum of the negative value
        xp = [kim[arg_max_neg,i], kim[arg_max_neg+select,i]]              #find the maximum of the negative value and minimum of the positive value
        fp = [S[arg_max_neg,i], S[arg_max_neg+select,i]]
        # plt.plot(krm[arg_max_neg,i], kim[arg_max_neg,i], ls=' ', marker='.', color='r')
        # plt.plot(krm[arg_max_neg+select,i], kim[arg_max_neg+select,i], ls=' ', marker='.', color='r')
        s_choose[i-ind_min] = np.interp(phi0, xp, fp)   #do the fitting
    # plt.hlines(phi0, -0.5, 300, color='r', ls = '-', linewidth =1.1)

    return kr_choose, s_choose

def cal_data(data_path, cal_name, fn_cal_data, pos_l2, pos_lphi, frq, **kwargs):
    """
    In experiment, we use Ps4 to change the length of lphi,
    use Ps2 to change the length of l2

    """
    data_path = 'D:/data_Marburg/' + data_path +'/'
    cal_path = 'D:/data_Marburg/cal_file/'+cal_name+'/'
    fn_cal_data = 'D:/data_Marburg/cal_data/'+fn_cal_data
    data = np.zeros((len(pos_l2), len(pos_lphi), len(frq)), complex)
    for m,j in enumerate(pos_l2):
        for n,i in enumerate(pos_lphi):
            filename = f'Ygraph_PS2={j}_PS4={i}.xmldat'
            freq1,data1=caldata(fn=[data_path+filename],
                                calname=cal_name,
                                calpath=cal_path)
            data[m,n,:] = data1[0,:,0]
    np.save(fn_cal_data, data, allow_pickle=True, fix_imports=True)
    return data

def fit_phaser():
    """
    We have the info of the length of phase shifter PS4, which
    is extracted from the optimization procedure, and we can use
    the fit parameter to interpole any PS4 position.

    """
    path0 = 'D:/data_Marburg/'
    fn_length = path0 + 'Cablelength/230526_Ygraph_bonds.npy'
    (_, phaser0) = np.load(fn_length, allow_pickle=True, fix_imports=True)
    pos_phaser0 = np.linspace(0, 200000, 401, dtype=np.int)
    fit_par = np.polyfit(pos_phaser0, phaser0, 1)
    return fit_par

def cal_real_phaser(pos_lphi, adjust_phaser=0):
    """
    Using the fit parameter to interpole any PS4(lphi) position.

    """
    fit_par = fit_phaser()
    phaser = pos_lphi*fit_par[0]+fit_par[1]+adjust_phaser
    return phaser

def estimate_zoom_info(k_zoom, phi_zoom):
    """
    If we want to do a zoom measurement, the set parameters of
    the measurement should be changed, and in this code, we can
    estimate the new frq range and position range of Ps2(l2)

    """
    l_zoom_ex = np.array([phi_zoom[1]/k_zoom[0]/2, phi_zoom[0]/k_zoom[1]/2])
    print('Frq = ' + str(np.array(k_zoom)/(2*np.pi/0.299792458)))
    fit_par = fit_phaser()
    print('Pos_ps2 = ' + str((l_zoom_ex-fit_par[1])/fit_par[0]))

def k_grid(k, phaser):
    """
    Use wavenumber k and the length of lphi, we can
    construct the k grid of the system, and kim0 corresponding
    the length grid of lphi, and kim corresponding the phase grid

    """
    kr = k
    ki = phaser
    krm, kim = np.meshgrid(kr, ki)
    kim0 = kim.copy()
    for i in range(len(phaser)):
        kim[i,:]=kim[i,:]*kr*2
    # plt.figure()
    # n = np.shape(krm)[0]*np.shape(krm)[1]
    # plt.plot(np.reshape(krm, n), np.reshape(kim, n), ls=' ', marker='.')
    return kr, ki, krm, kim, kim0

def plot_grid(k, phaser, S, xlim=None, ylim=None, flag_k_l=False, flag_k_phi=True):
    """
    Plot the figure of k-lphi and k-phi, the color represents
    the modula of S

    """
    kr, ki, krm, kim, kim0 = k_grid(k, phaser)
    plot_ind = np.linspace(0, len(kr)-1, 1001, dtype=np.int)
    kr = kr[plot_ind]
    krm = krm[:, plot_ind]
    kim = kim[:, plot_ind]
    kim0 = kim0[:, plot_ind]
    S = S[:, plot_ind]
    
    phi1 = 33*np.pi
    phi2 = 35*np.pi
    if flag_k_l:
        plt.figure()
        plt.pcolormesh(np.transpose(krm), np.transpose(kim0), 
                       np.transpose(np.abs(S)), shading='auto',
                       cmap=plt.cm.gray)
        # plt.plot(kr, phi1/kr/2, color='r', ls='-')
        # plt.plot(kr, phi2/kr/2, color='b', ls='-')
        plt.xlim(xlim)
        plt.ylim((np.min(ki), np.max(ki)))
        plt.xlabel(r'$k\,(\mathrm{m}^{-1})$')
        plt.ylabel(r'$\ell_\varphi\,(\mathrm{m})$')
        
    if flag_k_phi:
        plt.figure()
        plt.pcolormesh(np.transpose(krm), np.transpose(kim), 
                       np.transpose(np.abs(S)), shading='auto',
                       cmap=plt.cm.gray)
        # plt.hlines(phi1, -0.5, 300, color='r', ls = '-', linewidth =1.1)
        # plt.hlines(phi2, -0.5, 300, color='b', ls = '-', linewidth =1.1)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(r'$k\,(\mathrm{m}^{-1})$')
        plt.ylabel(r'$\varphi$')
 
def extract_sp_basic(S, k, phaser, phis):
    """
    Extract the spectrum according to the phis

    """
    kr, ki, krm, kim, kim0 = k_grid(k, phaser)
    save_data_all = []
    for j in range(len(S)):
        print(j)
        S1 = S[j]
        frq_sum = []
        data_sum = []
        for i in range(len(phis)):
            frq, data = new_spectrum_phi(kr, kim, phis[i], S1)
            frq_sum.append(frq)
            data_sum.append(data)
        save_data = [frq_sum, data_sum, phis]
        save_data_all.append(save_data)
    save_data_all = np.array(save_data_all, dtype=object)
    return save_data_all

def extract_sp(k, phaser, phis, fn_cal_data, fn_sp):
    """
    Extract the spectrum according to the phis

    """
    S = np.load(fn_cal_data, allow_pickle=True, fix_imports=True)
    save_data_all = extract_sp_basic(S, k, phaser, phis)
    np.save(fn_sp, save_data_all, allow_pickle=True, fix_imports=True)
    return save_data_all

def extract_poles_HI_basic(sp, var_trunc):
    poles = []
    for i in range(len(sp)):
        [frq_sum, data_sum, phi_sum] = sp[i]
        po = []
        print(i)
        for j in range(len(data_sum)):
            try:
                frq=frq_sum[j]
                data=data_sum[j]
                var_trunc = var_trunc
                f1=frq[0]
                f2=frq[-1]
                hi_get = hi.hi_base(frq, data, w1=f1,w2=f2,var_trunc=var_trunc,
                                    valmin=f1,valmax=f2,plot_flag=False, noline=True, 
                                    reflection_flag=True, MinAmpl=1e-5, PtError=0.1,
                                    phi = None)
                r = hi_get[1][:,0]
                r = r[(r.real>f1) & (r.real<f2)]
                po.append(r)
            except:
                po.append(np.array([np.nan]))
        poles.append(po)
    save_poles_info = (poles, phi_sum)
    save_poles_info = np.array(save_poles_info, dtype=object)
    return save_poles_info
    
def extract_poles_HI(fn_poles, fn_sp, var_trunc=60):
    sp = np.load(fn_sp, allow_pickle=True, fix_imports=True)
    save_poles_info = extract_poles_HI_basic(sp, var_trunc)
    np.save(fn_poles, save_poles_info, allow_pickle=True, fix_imports=True)
    return save_poles_info
     
def limit_meas(k, s, k_min, k_max):
    ind = np.where((k<k_max) & (k>k_min))
    return k[ind], s[ind]

def fit_two_Lorentz_detail(fit_guess, frq, S):
    fit_par_local = np.zeros((len(fit_guess), 6), dtype=complex)+np.nan+1j*np.nan
    fits_errors = np.zeros(len(fit_guess), dtype=complex)+np.nan+1j*np.nan
    for ii, fit in enumerate(fit_guess):
        if ~np.isnan(fit)[0]:
            try:
                fit_par_local[ii] = cFitting(S, frq, fit, 2, fit_func=cLorentz)
            except:
                pass
            if ~np.isnan(fit_par_local[ii])[0] & (np.abs(fit_par_local[ii][0])<200):
                fits_errors[ii] = np.sum(np.abs(np.angle(cLorentz(frq, fit_par_local[ii]))- np.angle(S)))
    fits_errors_notnan = fits_errors[~np.isnan(fits_errors)]
    fit_par_local_notnan = fit_par_local[~np.isnan(fits_errors),:]
    if len(fits_errors_notnan) != 0:
        fit_par = fit_par_local_notnan[np.argmin(np.abs(fits_errors_notnan))]
        return fit_par

def fit_two_Lorentz_basic(sp, fit_krange, fit_initial, double_fit=True):
    n = np.shape(sp)[0]
    m = np.shape(sp)[2]
    fit_pars = np.zeros((n, m, 6), complex)+np.nan+1j*np.nan
    for i in range(n):
        print(i)
        for j in range(m):
            frq, S = limit_meas(sp[i,0,j], sp[i,1,j], fit_krange[0], fit_krange[1])
            
            if i==0 and j==0:
                fit_guess=[fit_initial]
            elif i==0 and j!=0:
                fit_guess=[fit_pars[i,j-1], fit_initial]
            elif i!=0 and j==0:
                fit_guess=[fit_pars[i-1,j], fit_initial]
            else:
                fit_guess=[fit_pars[i,j-1], fit_pars[i-1,j]]
            
            fit_pars[i,j] = fit_two_Lorentz_detail(fit_guess, frq, S)
    
    if double_fit:
        inds_nan = np.argwhere(np.isnan(fit_pars[:,:,0]))
        for ind_nan in inds_nan[::-1]:
            i = ind_nan[0]
            j = ind_nan[1]
            frq, S = limit_meas(sp[i,0,j], sp[i,1,j], fit_krange[0], fit_krange[1])
            # plt.plot(frq,np.abs(S))
            
            if j==0 and 0<i<n-1:
                fit_guess=[fit_pars[i+1,j], fit_pars[i+1,j+1], 
                           fit_pars[i-1,j+1], fit_pars[i,j+1]] 
            elif j==m-1 and 0<i<n-1:
                fit_guess=[fit_pars[i+1,j], fit_pars[i+1,j-1], fit_pars[i-1,j-1]]
            elif 0<j<m-1 and 0<i<n-1:
                fit_guess=[fit_pars[i+1,j], fit_pars[i+1,j-1], fit_pars[i-1,j-1],
                           fit_pars[i+1,j+1], fit_pars[i-1,j+1], fit_pars[i,j+1]]
            elif 0<j<m-1 and i==0:
                fit_guess=[fit_pars[i+1,j], fit_pars[i+1,j-1], 
                           fit_pars[i+1,j+1],  fit_pars[i,j+1]]
            elif 0<j<m-1 and i==n-1:
                fit_guess=[ fit_pars[i-1,j-1], fit_pars[i-1,j+1], fit_pars[i,j+1]]
                
            fit_pars[i,j] = fit_two_Lorentz_detail(fit_guess, frq, S)
    return fit_pars
        
def fit_two_Lorentz(fn_poles, fn_sp, fit_krange, fit_initial, double_fit=True):
    sp = np.load(fn_sp, allow_pickle=True, fix_imports=True)
    fit_pars = fit_two_Lorentz_basic(sp, fit_krange, fit_initial, double_fit)
    np.save(fn_poles, fit_pars, allow_pickle=True, fix_imports=True)
    return fit_pars

def fit_two_Lorentz_initial_test(fn_poles, fn_sp, i, j, fit_krange, fit_initial):
    sp = np.load(fn_sp, allow_pickle=True, fix_imports=True)
    frq, S = limit_meas(sp[i,0,j], sp[i,1,j], fit_krange[0], fit_krange[1])
    fit_pars_test = cFitting(S, frq, fit_initial, 2, fit_func=cLorentz)
    return fit_pars_test
    
def plot_spectrum(sp, l2_ind=False, phase_ind=False):
    if l2_ind:
        n=(l2_ind, l2_ind+1)
    for i in range(n[0], n[1]):
        plt.figure()
        for j in phase_ind:
            plt.plot(sp[i,0,j], np.abs(sp[i,1,j])**2, label=f'phase_ind={j}')
        plt.title(f'l2_ind={i}')
        plt.xlabel(r'$\mathrm{Re}(k)$')
        plt.ylabel(r'$|S|^2$')
        plt.xlim((162,172))
        plt.ylim((-0.1,1.1))
        plt.legend()

def plot_poles_HI(poles, l2_ind=False, phase_ind=False):
    poles_plot = poles[0]
    if l2_ind:
        n=(l2_ind, l2_ind+1)
    for i in range(n[0], n[1]):
        po_plot = np.array([ii for k in poles_plot[i] for ii in k])
        plt.figure( figsize=(12, 4))
        # plt.clf()
        plt.plot(np.real(po_plot), np.imag(po_plot), ls=' ', marker='.', color='b')
        if phase_ind:
            plt.plot(np.real(poles_plot[i][phase_ind[0]]), 
                     np.imag(poles_plot[i][phase_ind[0]]), 
                     ls=' ', marker='o', color='r', 
                     markersize=6, label=f'phase_ind={phase_ind[0]}')
            plt.plot(np.real(poles_plot[i][phase_ind[1]]), 
                     np.imag(poles_plot[i][phase_ind[1]]), 
                     ls=' ', marker='s', color='g', 
                     markersize=6, label=f'phase_ind={phase_ind[1]}')
        plt.title(f'l2_ind={i}')
        plt.xlabel(r'$\mathrm{Re}(k)$')
        plt.ylabel(r'$\mathrm{Im}(k)$')
        # plt.xlim((162,172))
        plt.ylim((0,2))
        plt.legend()

def plot_poles_LZ(fit_pars, l2_ind=False, phase_ind=False):
    r1=fit_pars[:,:,1]
    r2=fit_pars[:,:,3]
    r1 = r1[(r1.real<200) & (r1.real>100)]
    r2 = r2[(r2.real<200) & (r2.real>100)]

    # plt.figure('EP')
    # plt.clf()
    plt.figure()
    plt.plot(np.real(r1), np.imag(r1), ls=' ', marker='.', color='orange')
    plt.plot(np.real(r2), np.imag(r2), ls=' ', marker='.', color='orange')
    
    if l2_ind:
        n=(l2_ind, l2_ind+1)

    for i in range(n[0], n[1]):
        plt.plot(np.real(fit_pars[i,:,3]), np.imag(fit_pars[i,:,3]), ls=' ', marker='.', color='r')
        plt.plot(np.real(fit_pars[i,:,1]), np.imag(fit_pars[i,:,1]), ls=' ', marker='.', color='r')
    
        if phase_ind:
            plt.plot(np.real(fit_pars[i,phase_ind[0],[1,3]]), 
                     np.imag(fit_pars[i,phase_ind[0],[1,3]]),
                     ls=' ', marker='o', color='b', 
                     markersize=6, label=f'phase_ind={phase_ind[0]}')
            plt.plot(np.real(fit_pars[i,phase_ind[1],[1,3]]), 
                     np.imag(fit_pars[i,phase_ind[1],[1,3]]), 
                     ls=' ', marker='s', color='g', 
                     markersize=6, label=f'phase_ind={phase_ind[1]}')
            plt.title(f'l2_ind={i}')
            plt.xlabel(r'$\mathrm{Re}(k)$')
            plt.ylabel(r'$\mathrm{Im}(k)$')
            plt.xlim((166.8,167.3))
            plt.ylim((0.3,0.85))
            plt.legend()

def plot_fit_check(sp, fit_pars, l2_ind, phase_ind, fit_k=np.linspace(166,168,1000)):
    plt.figure()
    plt.plot(sp[l2_ind,0,phase_ind], np.abs(sp[l2_ind,1,phase_ind])**2, label=f'phase_ind={phase_ind}')
    plt.plot(fit_k, np.abs(cLorentz(fit_k, fit_pars[l2_ind,phase_ind]))**2, label='fit curve')
    plt.title(f'l2_ind={l2_ind}')
    plt.xlabel(r'$\mathrm{Re}(k)$')
    plt.ylabel(r'$|S|^2$')
    plt.xlim((162,172))
    plt.ylim((-0.1,1.1))
    plt.legend()
    
def adjust_phaser_to_ex(fn_cal_data, k, phaser):
    
    S1 = np.load(fn_cal_data, allow_pickle=True, fix_imports=True)[-1]
    sum_sp = []
    add_phaser = np.linspace(0.0006, 0.001, 101)
    for i in range(len(add_phaser)):
        kr, ki, krm, kim, kim0 = k_grid(k, phaser+add_phaser[i])
        frq, data = new_spectrum_phi(kr, kim, 33*np.pi, S1)
        frq, data1 = new_spectrum_phi(kr, kim, 35*np.pi, S1)
        sum_sp.append(np.sum(np.abs(data)+np.abs(data1)))
        # plt.plot(frq,np.abs(data))
    print(add_phaser[np.argmax(sum_sp)])
        
def fit_pars_recover(fit_pars0, imag_bg):
    fit_pars = fit_pars0.copy()
    fit_pars[:,:,0] = -np.conj(fit_pars[:,:,0])
    fit_pars[:,:,1] = np.conj(fit_pars[:,:,1]) + imag_bg*1j
    fit_pars[:,:,2] = -np.conj(fit_pars[:,:,2])
    fit_pars[:,:,3] = np.conj(fit_pars[:,:,3]) + imag_bg*1j
    fit_pars[:,:,4] = -np.conj(fit_pars[:,:,4])
    fit_pars[:,:,5] = -np.conj(fit_pars[:,:,5])
    return fit_pars
    
    
    