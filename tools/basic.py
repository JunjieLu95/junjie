# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:20:09 2024

@author: jlu
"""

import os
import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
from mesopylib.extract.cLorentzFit import cFitting,cLorentz

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def single_fit(S, frq):
    '''
    Use Lorentz fitting to fit the isolated resonances/zeros,
    the function can skip the initial gussing and remove background,
    which helps us performing fitting quickly

    Parameters
    ----------
    S : array
        complex spetrum.
    frq : array
        corresponding frqs.

    Returns
    -------
    z : list
        the information of extracted resonance.

    '''
    frq = np.real(frq)
    
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
    return z

def get_lim(rmt_array, lim_range, which_complex):
    '''
    Give an array, and pick the values you want to restrict.
    
    Parameters
    ----------
    rmt_array : array
        rmt full array.
    lim_range : tuple
        for example, the range can be (-10, 10)
    which_complex : str
        'real' or 'imag'
    '''
    
    if which_complex=='real':
        values = np.real(rmt_array)
    if which_complex=='imag':
        values = np.imag(rmt_array)
    lim = (values>lim_range[0]) & (values<lim_range[1])
    return rmt_array[lim]

def Gauss_standard(x, mu, sigma):
    """
    Gaussian curve, with the size = 1

    """
    y=1/(sigma * np.sqrt(2 * np.pi)) *np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

def Gauss(x, a, mu, sigma):
    """
    Gaussian curve

    """
    y= a *np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

def fit_Gauss(xydata, func, range_fit=None):
    """
    Fit the data to Gaussian curve

    """
    
    xdata, ydata = xydata
    if range_fit is not None:
        mask = (xdata >= range_fit[0]) & (xdata <= range_fit[1])
        xdata = xdata[mask]
        ydata = ydata[mask]
    else:
        mask = slice(None)
        
    parameters, covariance = curve_fit(func, xdata, ydata)
    
    new_x = np.linspace(xydata[0][0], xydata[0][-1], 501)
    fit_y = func(new_x, *parameters)
    
    return new_x, fit_y, parameters

def exponential(x, B):
    """
    Exponential curve

    """
    y = 1/2/B*np.exp(-1/B*x)
    return y

def fit_exponential(xydata):
    """
    Fit the hist to Exponential curve

    """
    x_hist = xydata[1]
    y_hist = xydata[0]
    newx = (x_hist[:-1]+x_hist[1:])/2
    N = len(newx)
    x_sum = np.insert(newx[int((N+1)/2):],0,0)
    y_sum = np.insert(np.flip(y_hist[:int((N-1)/2)]) + y_hist[int((N+1)/2):], 
                      0, 2*y_hist[int((N-1)/2)])
    parameters, covariance = curve_fit(exponential, x_sum, y_sum/2)
    fit_A = parameters[0]
    fit_y = exponential(x_sum, fit_A)
    return x_sum, fit_y

def sine_model(x, A, k, theta, c):
    return A * np.sin(k * x + theta) + c

def target_model(x, a, b, c, A, k, theta, C):
    return (a * x**2 + b * x + c) * sine_model(x, A, k, theta, C)

def fit_time_temperature(x_data, y_data):
    fft_y = np.fft.fft(y_data - np.mean(y_data)) 
    fft_freq = np.fft.fftfreq(len(x_data), (x_data[1] - x_data[0]))
    guess_freq = abs(fft_freq[np.argmax(abs(fft_y[1:])) + 1])
    
    initial_guess_sine = [np.std(y_data), 2. * np.pi * guess_freq, 0, np.mean(y_data)]
    popt_sine, pcov_sine = curve_fit(sine_model, x_data, y_data, p0=initial_guess_sine)
    
    A_sine, k_sine, theta_sine, c_sine = popt_sine
    initial_guess_target = [0, 0, 0, A_sine, k_sine, theta_sine, c_sine] 
    
    popt_target, pcov_target = curve_fit(target_model, x_data, y_data, p0=initial_guess_target)
    
    temp_smooth = target_model(x_data, *popt_target)
    return temp_smooth
