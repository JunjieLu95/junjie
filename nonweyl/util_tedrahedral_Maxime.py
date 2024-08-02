# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:18:14 2021

@author: Junjie
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from junjie.tools import single_fit
from mesopylib.extract.cLorentzFit import cFitting,cLorentz


def creat_rand_bonds(set_ratio, flag_extreme):
    '''
    Create different tedrahedral graph realizations for calculate the poles.
    One can set the ratio between the minimum bond length around the vertex connected with danling bond
    and the length of dangling bond l_min/l_d.
    To be noticed, for simplicity, the length of dangling bond is always fixed and once the ratio is not equal 
    to 1, we choose lengths[1] and lengths[5] randomly as l_min=set_ratio*l_d.
    
    Parameters
    ----------
    set_ratio : float
        the ratio should not larger than one.

    Returns
    -------
    lengths : array
        the length of the graph.
    
    '''
    bonds_origin = np.array([0.94939222, 0.37434459, 1.75148134, 1.59136089, 
                             0.86751784, 0.78565432, 0.43766953, 0.24352648, 
                             0.32015181])
    ratio = 233 # start ratio, means nothing
    if set_ratio>1:
        print('Not possible.')
    else:
        while ratio!=set_ratio:
            lengths = np.zeros((8))
            lengths[7] = bonds_origin[7]
            if set_ratio == 1:
                t_length = np.sum(bonds_origin[:8])-np.sum(lengths)
                fences = np.zeros((8))
                fences[1:-1] = np.random.random((6))
                fences[-1] = 1
                lengths_temp = np.diff(np.sort(fences))*t_length
            else:
                inds = [1,5]
                if flag_extreme:
                    for i in inds:
                        lengths[i] = bonds_origin[7]*set_ratio
                    
                    t_length = np.sum(bonds_origin[:8])-np.sum(lengths)
                    fences = np.zeros((6))
                    fences[1:-1] = np.random.random((4))
                    fences[-1] = 1
                    lengths_temp = np.diff(np.sort(fences))*t_length
                else:
                    select_ind = inds[np.random.randint(0, 2, 1)[0]]
                    lengths[select_ind] = bonds_origin[7]*set_ratio
                    
                    t_length = np.sum(bonds_origin[:8])-np.sum(lengths)
                    fences = np.zeros((7))
                    fences[1:-1] = np.random.random((5))
                    fences[-1] = 1
                    lengths_temp = np.diff(np.sort(fences))*t_length
        
            lengths[lengths==0] = lengths_temp
            l_min = np.min([lengths[1], lengths[5], lengths[7]])
            ratio = l_min/lengths[7]
        return lengths
    
def graph_smatrix_bond_matrix(l, wr, k, openT, phi0):
    """ 
    Function for calculating the S-matrix, used for function 'cal_s'
    """
    
    ndim = len(l)
    nkmax = len(k)
    c = np.zeros((ndim, ndim))
    c[np.nonzero(l)]=1 
    c[5,6] = 1
    c[6,5] = 1
    ndim1 = int(np.sum(c)/2)
    
    ports = wr
    v = np.zeros(ndim)+3
    v[6]=1
    rho = np.zeros(ndim)+1
    for i in range(len(ports)):
        rho[ports[i]]=2/v[ports[i]]-1
    tau = np.zeros((ndim, ndim))
    for i in range(len(ports)):
        for j in range(ndim):
            tau[ports[i], j]=2/v[ports[i]]
    
    dd = np.zeros((nkmax, (2*ndim1)**2), dtype = 'complex_')
    nn1=-1
    for i1 in range(ndim):
        for j1 in range(ndim):
            if c[i1,j1]!=0:
                nn1+=1
                phi=k*l[i1,j1]
                if openT==True:
                    if i1==6 and j1==5:
                        phi=phi+phi0
                    if i1==5 and j1==6:
                        phi=phi+phi0
                dd[:,nn1*(2*ndim1+1)]=np.exp(1j*phi)
    dd=np.reshape(dd, (nkmax, 2*ndim1, 2*ndim1))            
                
    nn1=-1
    ss = np.zeros((2*ndim1, 2*ndim1), dtype = 'complex_')
    for i1 in range(ndim):
        for j1 in range(ndim):
            if c[i1,j1]!=0:
                nn1+=1
                nn2=-1
                for i2 in range(ndim):
                    for j2 in range(ndim):
                        if c[i2,j2]!=0:
                            nn2+=1
                            if j1==i2:
                                ss[nn1,nn2]=2/v[j1]
                                if i1==j2:
                                    ss[nn1,nn2]=2/v[j1]-1
                            if j1==6:
                                if j1==i2:
                                    ss[nn1,nn2]=0
                                    if i1==j2:
                                        ss[nn1,nn2]=1

    tt = ss+0
    
    sb = np.einsum('nij,jk->nik', dd, tt)
    zinv = np.linalg.inv(np.eye(2*ndim1)-sb)  
    smat = np.zeros((nkmax,ndim,ndim), dtype = 'complex_')
    for ii in range(len(ports)):
        smat[:,ports[ii],ports[ii]]=rho[ports[ii]]
    nn1 = -1
    for i1 in range(ndim):
        for j1 in range(ndim):
            if c[i1,j1]!=0:
                nn1+=1
                nn2=-1
                for i2 in range(ndim):
                    for j2 in range(ndim):
                        if c[i2,j2]!=0:
                            nn2+=1
                            smat[:,i1,j2]=smat[:,i1,j2]+tau[i1,j1]*zinv[:,nn1,nn2]*dd[:,nn2,nn2]*tau[j2,i2]
    
    nonz = np.nonzero(smat)
    S = smat[nonz[0], nonz[1], nonz[2]]
    S = np.reshape(S, (nkmax, len(ports), len(ports)))
    return S

def cal_s(l, phi, k_guess, ss=1):
    """
    Calculate the spectrum for the considered tedrahedral graph (wr = [5]),
    with k_guess, we roughly know where the poles are,
    so we set a frq range with 2*ss length, then we can get the spectrum
    
    PS: We need to make sure there are enough points in the resonance peak
    and the peak is almost in the center. 
    Tip: When dealing the case that the peak of resonance is too sharp, 
    it is better to adjust the imag part of frq.

    """
    shift_imag = 0.07j
    k = np.linspace(np.real(k_guess)-ss, np.real(k_guess)+ss, 5000) + 1j*np.imag(k_guess) + shift_imag
    S = graph_smatrix_bond_matrix(l, [5], k, True, phi/2)[:,0,0]
    S11a = np.abs(S)
    S0 = np.copy(S11a)
    k_max = k[np.argmax(S11a)] 
    n_loop = 0
    while np.max(S11a)<100 or len(np.where(S11a>np.max(S11a)*2/3)[0])<50 or np.argmax(S11a)<600 or np.argmax(S11a)>4400:
        k = np.linspace(k_max-ss, k_max+ss, 5000) + shift_imag
        S = graph_smatrix_bond_matrix(l, [5], k, True, phi/2)[:,0,0]
        S11a = np.abs(S)
        diff_S = np.max(S11a) - np.max(S0)
        if diff_S<0:
            shift_imag = -shift_imag
        k_max = k[np.argmax(S11a)]
        # print(np.max(S11a))
        n_loop+=1
        if n_loop>40:
            break
    return k, S

def cal_poles_cLorentz(ratios, phis, n_realizations, n_frqs, fn, flag_extreme=False):
    """
    Return the numerical results and prediction results of poles.

    Parameters
    ----------
    ratios : array
        the ratio between the minimum bond length around the vertex connected 
        with danling bond and the length of dangling bond
    phis : array
        phis array extremely closed to 0 or 2*pi
    n_realizations : int
        the number of graph realizations with different length, but same ratios.
    n_frqs : int
        the number of poles transitions in different regions, 
        if n_frqs=1 and phis are closed to 0 and 2*pi, 
        there are 2 transitions of poles going to infinite.

    """
    n_ratios = len(ratios)
    n_phis = len(phis)

    k_predict = np.zeros((n_ratios, n_realizations, n_frqs, n_phis), dtype=complex)
    k_numeric = np.zeros((n_ratios, n_realizations, n_frqs, n_phis), dtype=complex)
    l_all = np.zeros((n_ratios, n_realizations, 7, 7))
    
    fit_erro = []
    theDict={}
    for ir in range(n_ratios):
        print(f'ratio {ratios[ir]}')
        for i in range(n_realizations):
            print(f'realization {i}')
            ndim = 7
            l = np.zeros((ndim, ndim))
            b = creat_rand_bonds(ratios[ir], flag_extreme)
            
            l[0,1] = b[0]
            l[0,3] = b[1]
            l[0,4] = b[2]
            l[1,2] = b[3]
            l[1,4] = b[4]
            l[2,3] = b[5]
            l[2,4] = b[6]
            l[3,5] = b[7]
            l[5,6] = 0
            l = l + l.T
            l_all[ir, i] = l
            
            b_dangling = l[3,5]
            k_dangling = np.pi/b_dangling
            
            for j in range(n_frqs):
                for m in range(n_phis):
                    phi = phis[m]
                    k = 1j/2/b_dangling*np.log(3j/2*np.tan(phi/2)) + k_dangling*(1/2)*(2*j+1)
                    k_predict[ir, i, j, m] = k
                    
                    try:
                        frq, S = cal_s(l, phi, k_guess=k, ss=1)
                        plt.plot(frq.real, np.abs(S))
                        k_rough = single_fit(S, frq)[1] + np.imag(frq[0])*1j
                        plt.plot(frq.real, np.abs(cLorentz(frq.real,  single_fit(S, frq))))
                        
                        frq, S = cal_s(l, phi, k_rough, ss=1)
                        k_num = single_fit(S, frq)[1] + np.imag(frq[0])*1j
                        
                        k_numeric[ir, i, j, m] = k_num
                        
                        if np.isnan(k_num) or np.real(k_num)>300 or np.real(k_num)<-300:
                            fit_erro.append((ir, i, j, m))
                            print((ir, i, j, m))
                            print('fit error')
                    except:
                        k_numeric[ir, i, j, m] = np.nan   
                        fit_erro.append((ir, i, j, m))
                        
    
        theDict["ratios"] = ratios
        theDict["phis"] = phis
        theDict["n_realizations"] = n_realizations
        theDict["n_frqs"] = n_frqs
        theDict["graph"] = l_all
        theDict["k_predict"] = k_predict
        theDict["k_numeric"] = k_numeric
        theDict["fit_erro"] = fit_erro
        
        with open(fn, "wb") as f:
            pickle.dump(theDict, f)  
    
    return theDict

def cal_radius(phis, l_all, n_realizations, n_frqs, ratios):
    """
    Return the radius of disks.
    Here we choose \eplison=0, and \theta=1/3.

    Parameters
    ----------
    phis : array
        phis array extremely closed to 0 or 2*pi
    l_all : array
        length matrix of graph
        
    """
    disk_all = np.zeros((len(ratios), n_realizations, n_frqs, len(phis)))
    for i in range(len(ratios)):
        for j in range(n_realizations):
            l = l_all[i, j]
            l_min = np.min([l[3,5], l[0,3], l[2,3]])
            l_b = l[3,5]
            for k in range(len(phis)):
                phi = phis[k]
                disk_r = 3/2/l_b*(3/2*np.abs(np.tan(phi/2)))**(l_min/l_b)
                disk_all[i,j,:,k] = np.real(disk_r)
    return disk_all
    
    