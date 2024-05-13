# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:53:08 2024

@author: jlu
"""

import os
import platform
import numpy as np

computer_name=platform.node()

if computer_name=='pwluj0041':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/2021-11-03-RMT_Zeros-extract/data/'
elif computer_name=='LAPTOP-K2TVHSG9':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/2021-11-03-RMT_Zeros-extract/data/'
elif computer_name=='PHYS-PC1TJA6-3B':
    path_base = 'C:/Junjie/data/'
    
def format_kappa(kappa):
    if isinstance(kappa, float) and kappa.is_integer():
        return str(int(kappa))
    elif isinstance(kappa, int):
        return str(kappa)
    kappa_rounded = np.around(kappa, 3)
    if kappa_rounded % 1 == 0 or kappa_rounded * 1000 % 10 == 0:
        return "{:.4f}".format(kappa)
    else:
        return "{:.3f}".format(kappa_rounded)

def fn_save_Ta(path, N, M, nReal, kappa, E):
    kappa_str = format_kappa(kappa)
    path += 'data_Ta/'
    if not os.path.exists(path):
        os.makedirs(path)
    if len(E)==1:
        fn = path + f'Ta_N={N}_M={M}_nReal={nReal}_kappa_{kappa_str}_ERe{int(np.real(E[0]))}_EIm{np.imag(E[0])}.npy'
    return fn

def fn_save_super(path, N, M, kappa, n_lambda, n_dys, Im_range, E_range=None):
    kappa_str = format_kappa(kappa)
    if E_range:
        file_prefix = f"super_poles_dynamics_N={N}_M={M}_kappa_{kappa_str}_nlamda={n_lambda}_nReal={n_dys}_start_{Im_range[0]}_{Im_range[-1]}"
        file_prefix += f'_Erange_{E_range[0]}_{E_range[1]}'
        fn = path + 'Plot_' + file_prefix + '.npy'
        fn_info = path + 'Plot_' + 'info_' + file_prefix + '.npy'
    else:
        path += f'data_dynamics/super/N_{N}_{kappa_str}/'
        if not os.path.exists(path):
            os.makedirs(path)
        file_prefix = f"super_poles_dynamics_N={N}_M={M}_kappa_{kappa_str}_nlamda={n_lambda}_nReal={n_dys}_start_{Im_range[0]}_{Im_range[-1]}"
        fn = path + file_prefix + '.npy'
        fn_info = path + 'info_' + file_prefix + '.npy'
    return fn, fn_info

def fn_save_narrow(path, N, M, kappa, n_lambda, n_dys, flag_Plot):
    kappa_str = format_kappa(kappa)
    if flag_Plot:
        fn = path+f'Plot_Ysmall_poles_dynamics_N={N}_M={M}_kappa_{kappa_str}_nlamda={n_lambda}_nReal={n_dys}.npy'
    else:
        path += f'data_dynamics/narrow/N_{N}_{kappa_str}/'
        if not os.path.exists(path):
            os.makedirs(path)
        fn = path+f'Ysmall_poles_dynamics_N={N}_M={M}_kappa_{kappa_str}_nlamda={n_lambda}_nReal={n_dys}.npy'
    return fn

def fn_save_ei_Heff(path, N, M, kappa, looptime):
    kappa_str = format_kappa(kappa)
    path += 'data_ei/'
    if not os.path.exists(path):
        os.makedirs(path)
    fn = path+f'Poles_Heff_M{M}_N{N}_loop{looptime}_kappa{kappa_str}.npy'
    return fn

def fn_save_extract_zero(path, N, M, kappa, n_real, E):
    kappa_str = format_kappa(kappa)
    path += f'data_zero/N{N}/'
    if not os.path.exists(path):
        os.makedirs(path)
    fn = path+f'Zero_S_N={N}_M={M}_nReal={n_real}_kappa_{kappa_str}_EIm{np.imag(E[0])}.npy'
    return fn


