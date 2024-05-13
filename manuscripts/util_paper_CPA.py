#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:02:59 2021

@author: kuhl
"""
import platform
import matplotlib.pyplot as plt

computer_name=platform.node()

if computer_name=='plukh0041':
    path_base = '/home/kuhl/tmp/'
elif computer_name=='pwluj0041':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/2021-11-03-RMT_Zeros-extract/'
elif computer_name=='LAPTOP-K2TVHSG9':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/2021-11-03-RMT_Zeros-extract/'
    
path_data = path_base + 'data/manuscript/'
path_data0 = path_base + 'data/'
path_figs = path_base + '2023_Superradiant_Coherent_Perfect_Absorption/figs/'

path_JNM_data = path_base + 'manuscript/2024_JNM/data/'
path_JNM_figs = path_base + 'manuscript/2024_JNM/Figures/'

def useTex(_useTex, xs=15, ys=None):
    #xs=15, size=12 for phd thesis
    #
    if ys is None:
        ys=xs/((1+5**.5)/2)
    inch2cm=2.54
    if _useTex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=12)
        plt.rc('figure', figsize= [xs/inch2cm, ys/inch2cm])
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')
        plt.rc('font', size=10)
        plt.rc('figure', figsize= [xs, ys])
        

