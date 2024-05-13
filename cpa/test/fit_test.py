# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:39:12 2022

create first main figure for the publication of CPA

@author: Junjie
"""
import numpy as np
import matplotlib.pyplot as plt
from mesopylib.utilities.load_save.loadXmlDat import loadXmlDat
from mesopylib.extract.cLorentzFit import cFitting,cLorentz

fn1 = 'fit_test_data.xmldat'

d=loadXmlDat(fn1)
frq_rand = d[0][0].getFrqAxis()
S=d[1]
exp_S_rand = np.reshape(S, (4, 4, S.shape[1]), order='F')
exp_S_rand = np.transpose(exp_S_rand,(2,0,1))

frq=frq_rand
S=exp_S_rand[:,0,1]
fp = [frq[0], frq[-1]]
sp = [S[0], S[-1]]
background = [fp, sp]
    
pr = np.argmax(np.abs(S))
ifrq = frq[pr]
amp = S[pr]
bg_fit = np.abs(np.interp(ifrq, background[0], background[1]))

find_width = np.abs(np.abs(S) - (np.abs(amp)+bg_fit)/2)
find_width_arg = find_width.argmin()
width = np.abs((frq[find_width_arg] - ifrq)/2)
width1 = -width

pc_init=[amp*width*np.exp(-1j*np.pi/2), ifrq+width*1j, 0.00+0.00j, 0.0+0.0j]
pc_init1=[amp*width1*np.exp(-1j*np.pi/2), ifrq+width1*1j, 0.00+0.00j, 0.0+0.0j]

s1 = np.sum(np.abs(np.angle(cLorentz(frq, pc_init))- np.angle(S)))
s2 = np.sum(np.abs(np.angle(cLorentz(frq, pc_init1))- np.angle(S)))

try:
    if s1<s2:
        fit_result=[]
        fit_result_=cFitting(S, frq, pc_init, 1, fit_func= cLorentz)
        fit_result.append(fit_result_)
    else:
        fit_result=[]
        fit_result_=cFitting(S, frq, pc_init1, 1, fit_func= cLorentz)
        fit_result.append(fit_result_)
except:
    pass
z = fit_result_

plt.figure()
plt.plot(frq_rand, np.abs(S))
plt.plot(frq_rand, np.abs(cLorentz(frq, pc_init1)))
plt.plot(frq_rand, np.abs(cLorentz(frq, pc_init)))

plt.figure()
plt.plot(frq_rand, np.abs(S))
plt.plot(frq_rand, np.abs(cLorentz(frq, z)))
