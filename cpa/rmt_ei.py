# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:45:11 2021

@author: Junjie
"""

import numpy as np
import mesopylib.num.rmt.rmt_num as r
from junjie.cpa.saving import fn_save_ei_Heff

def cal_Heff_ei(M, N, kappa, looptime, path='', flag_save=True):
    """
    calculate the eigenvalues of effective Heff

    Parameters
    ----------
    M : int
        number of channels
    N : int
        size of Heff
    kappa : float
        coupling strength.
    looptime : int
        number of realizations
    path : str
        path to save
    flag_save : bool
        save or not. The default is True.

    Returns
    -------
    ei_heff_sum_ar : array
        DESCRIPTION.

    """
    ei_heff_sum = []
    for i in range(looptime):
        heff = r.createRmtH_with_opening(M, N, kappa, beta=1,
                                    seed_H=None,
                                    random_container_V=np.random,
                                    return_V=False)
        ei_heff = np.linalg.eigvals(heff)
        ei_heff_sum.append(ei_heff)
        
    ei_heff_sum_ar = np.array([i for k in ei_heff_sum[:] for i in k])
    
    if flag_save:
        fn = fn_save_ei_Heff(path, N, M, kappa, looptime)
        np.save(fn, ei_heff_sum_ar, allow_pickle=True, fix_imports=True)
        
    return ei_heff_sum_ar


# M = 4
# N = 20
# looptime=40000
# kappa = 3.441
# cal_Heff_ei(M, N, kappa, looptime)

