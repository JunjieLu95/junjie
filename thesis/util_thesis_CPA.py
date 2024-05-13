#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:02:59 2021

@author: kuhl
"""
import platform
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from junjie.cpa.extract import Norm_complex_ei_3d_cavity

computer_name=platform.node()

if computer_name=='plukh0041':
    path_base = '/home/kuhl/tmp/'
elif computer_name=='pwluj0041':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/mythesis/chapterCPA/'
elif computer_name=='LAPTOP-K2TVHSG9':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/mythesis/chapterCPA/'
    
path_data = path_base + 'data/'
path_figs = path_base + 'figs/'

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

def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def target(x, a, b, c, d, e, f, g):
    """
    degree 2 polynomial * sine
    """
    return (a*x**2+b*x+c)*(d*np.sin(e*x+f)+g)

def plot_average(path_ex, fn, which, label_name, ax):
    p_xs = []
    p_ys = []
    ns = []
    for i in range(len(fn)):
        zero_ex = load_poles_zeros(path_ex, fn[i], which=which)
        zero_ex = Norm_complex_ei_3d_cavity(zero_ex)
        zero_ex = get_lim(zero_ex, (-53, 53), 'imag')
        ns.append(len(zero_ex))
        p_y, p_x = np.histogram(zero_ex.imag, density=True, bins=200, range=(-53, 53))
        p_xs.append(p_x)
        p_ys.append(p_y)
    p_xs = np.array(p_xs)
    p_ys = np.array(p_ys)
    ax.plot(np.mean(p_xs, axis=0)[:-1], np.mean(p_ys, axis=0), label = label_name+f', n={int(np.mean(ns))}')

def load_poles_zeros(path, fn, which):
    if which=='poles':
        rmt_results = np.conj(np.load(path + fn, allow_pickle=True, fix_imports=True))
    if which=='zeros':
        rmt_results = np.load(path + fn, allow_pickle=True, fix_imports=True)
    return rmt_results

def plot_histogram(rmt_poles, poles_complex, label, bins=501, range=None, color=None, ls=None, ax=None):
    if poles_complex=='imag':
        p_rmt_poles = np.histogram(rmt_poles.imag,  bins=bins, density=True, range=range)
        plt.xlabel(r'$\mathrm{Im}(E)$')
        plt.ylabel(r'$P(\mathrm{Im}(E))$')
    if poles_complex=='real':
        p_rmt_poles = np.histogram(rmt_poles.real,  bins=bins, density=True, range=range)
        plt.xlabel(r'$\mathrm{Re}(E)$')
        plt.ylabel(r'$P(\mathrm{Re}(E))$')
    ax.plot(p_rmt_poles[1][:-1], p_rmt_poles[0], label=label, color=color, ls=ls)  

    # plt.legend()
    return p_rmt_poles
