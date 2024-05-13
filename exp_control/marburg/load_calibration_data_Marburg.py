#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:04:43 2021

@author: kuhl
"""
# non_wely_motor_pos_0.xmldat
# non_wely_motor_pos_500.xmldat
# ....
# non_wely_motor_pos_200000.xmldat
# 
import numpy as np
import matplotlib.pyplot as plt
# path='D:/data/dataMR/Non_Weyl_0601/measurements/01062021/ps/'
#path='z:/raw/exp/graphs/non-weyl/20210421/'
path='D:/data/dataMR/23092021/lasso_2_3/'
name="lasso_2_3_{}.xmldat"
from mesopylib.utilities.load_save.loadXmlDat import loadXmlDat

fn=[path+name.format(i) for i in range(0,200000+1,500)]
#fn=fn[0:3]
from mesopylib.extract.errorTerms import calcErrTerms,calibration
fnpath='D:/data/dataMR/lasso_2_3/cal20092021/'
name='cal15092021'
err_terms=calcErrTerms(path=fnpath, name=name,cal_kit='VNA Agilent8720ES')
DATA=loadXmlDat(fn,SepWindows=True)
frqcal=DATA[0][0].getFrqAxis()*1e9    ### in [GHz]
freq=np.reshape(frqcal,-1)
nwindows=20
npointsperwindow=1601
index=np.arange(1,nwindows,1)*npointsperwindow
freq=np.delete(freq,index)
data=np.zeros((len(fn),len(freq)),dtype=complex)
for i in range(len(fn)):
    DATAcal=calibration((DATA[0],DATA[1][...,i]),err_terms)
    data[i]=np.delete(DATAcal[0],index)

plt.plot(freq,np.abs(data[0,:]))
plt.plot(freq,np.abs(data[1,:]))
plt.plot(freq,np.abs(data[100,:]))

np.save(f'Data_Tobias_Lasso_2_3_0923.npy', data, allow_pickle=True, fix_imports=True)
