# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:08:57 2022

@author: jlu
"""
import numpy as np
import matplotlib.pyplot as plt
from mesopylib.num.rmt.rmt_th import rmt2beta, wigner, th_P_S_Table

def th_Ps(x, beta=1, Poisson=None, GOE=None, GUE=None, GSE=None, wignerFlag=True, taylorFlag=False):
    """ returns the nearest neighbour spacing distribution in the Wigner approximation\n
        or as exact result using Taylor series and Pade approximants\n
        (see B. Dietz, F. Haake: Z. Phys. B80, 153 (1990) 'Taylor and Padé Analysis of the Level Spacing Distributions of Random-Matrix Ensembles' and
        F.J. Dyson: Commun. Math. Phys. 47, 171-183 (1976) 'Fredholm determinants and inverse scattering problems'
        wignerFlag = True -> gives Wigner back\n
        wignerFlag = False and tayorFlag=False -> gives Pade approximation\n
        wignerFlag = False and tayorFlag=True -> gives Taylor approximation\n

    """
    beta= rmt2beta(beta=beta, Poisson=Poisson, GOE=GOE, GUE=GUE, GSE=GSE)
    if wignerFlag:
        return wigner(x, beta=beta)

    if taylorFlag:
        m,a_m=th_P_S_Table(beta=beta,taylor=True)
        result=np.zeros(len(x), dtype=np.float)
        for i in np.arange(len(x)):
            result[i]=np.sum(a_m*x[i]**m)
        w=np.where(result <0)
        wCount=len(w)
        if wCount > 0:
            result[w]=0
        return result

    # pade 
    aproximants=th_P_S_Table(beta,pade=True)
    m1=aproximants['m_nu']
    nu=aproximants['nu_m']
    m2=aproximants['m_delta']
    delta=aproximants['delta_m']
    result=np.zeros(len(x), dtype=float)
    b=-0.219250583 
    #constant b is 1./24.*np.log(2)+3./2.*xsi_prime(-1)
    #obtained from:
    #Fredholm determinants and inverse scattering problems
    #https://doi.org/10.1007/BF01608375
    
    if beta==2:
        E_as=(np.pi/2.)**(-0.25)*np.exp(2*b)*x**(-0.25)*np.exp(-np.pi**2/8*x**2)
        P_as=np.pi**4/16.*(x**2-2./np.pi**2+5./np.pi**4/x**2)*E_as
        for i in np.arange(len(x)):
            result[i]=np.sum(nu*x[i]**(m1/4.))/np.sum(delta*x[i]**(m2/4.))
    if beta==1:
        C=2**(3/8)
        E_as=C*np.exp(b)*(np.pi*x+1)**(-1/8)*np.exp(-np.pi**2/16*x**2-np.pi/4*x)
        D=-1/8        
        P_as=np.pi**2/64*(64*D**2-16*np.pi**2*D*x**2-48*np.pi*D*x-96*D+np.pi**4*x**4+6*np.pi**3*x**3+5*np.pi**2*x**2-4*np.pi*x-4)/(np.pi**2*x**2+2*np.pi*x+1)*E_as
        numerator=np.zeros(len(x), dtype=float)
        denumerator=np.zeros(len(x), dtype=float)+delta[0]
        x=x*np.pi+1     # x=np.pi*s+1
        for i in np.arange(len(x)):
            mm=0
            for m in range(0,len(m1),2):
                mm+=1
                numerator[i]+=nu[m]*x[i]**(mm-7/8)+nu[m+1]*x[i]**(mm)
            mm=0
            for m in range(1,len(m2),2):
                mm+=1
                denumerator[i]+=delta[m]*x[i]**(mm-1/8)+delta[m+1]*x[i]**(mm)
        result=numerator/denumerator
    return result*P_as

def taylor_pade_goe(x):
    '''
    Returns the exact P(s) for GOE,
    for the small s, it is taken from taylor approximants,
    and for the large s, it is taken from pade approximants.
    
    '''
    taylor_goe = th_Ps(x, beta=1, Poisson=None, GOE=True, GUE=None, GSE=None, wignerFlag=False, taylorFlag=True)
    pade_goe = th_Ps(x, beta=1, Poisson=None, GOE=True, GUE=None, GSE=None, wignerFlag=False, taylorFlag=False)
    
    x_i0 = np.argmin(np.abs(x-2))
    x_i1 = np.argmin(np.abs(x-3))
    comb_index = np.argmin((pade_goe-taylor_goe)[x_i0:x_i1])+x_i0
    comb_goe = [j for i in [taylor_goe[:comb_index], pade_goe[comb_index:]] for j in i]
    return comb_goe

# x=np.linspace(0,7,2000)
# comb_goe = taylor_pade_goe(x)
# wigner_goe = th_Ps(x, beta=1, Poisson=None, GOE=True, GUE=None, GSE=None, wignerFlag=True, taylorFlag=False)

# plt.figure()
# plt.plot(x, comb_goe, label='taylor+pade')
# plt.plot(x, wigner_goe, label='wigner')
# plt.legend()
# plt.figure()
# plt.plot(x, comb_goe-wigner_goe, label='difference')
# plt.xlim(0,3.5)
# plt.ylim(-0.02,0.02)