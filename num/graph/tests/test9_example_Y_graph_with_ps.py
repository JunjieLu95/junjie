# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:11:45 2023

@author: jlu
"""
import numpy as np
import matplotlib.pyplot as plt
from mesopylib.extract.frq2 import frq2k
from scipy.optimize import minimize
from junjie.graph.num.smatrix import effective_h
from junjie.graph.num.analytical import Y_graph
import time

start = time.time()
for i in range(1):
    b = np.zeros((4))
    b[0] = 0.6317
    b[1] = 0.3183
    b[2] = 0.2230
    b[3] = 0.3423985890023513
        
    ndim = 5
    l = np.zeros((ndim, ndim))
    l[0,2] = b[0]
    l[1,2] = b[1]
    l[2,3] = b[2]
    l[3,4] = b[3]
    l = l + l.T
    nopen=1
    wr = np.zeros((nopen, ndim))
    wr[0,3] = 1
    frq = np.linspace(0.1, 18, 64001)
    k=frq2k(frq)*1e9
    kr = k.real
    
    k=kr+1j*0.09
    S = effective_h(l, wr, k, flag_eigenvalues=False)
end = time.time()
print(end - start)
    
start = time.time()
for i in range(1):
    S1 = Y_graph(k, b[0], b[1], b[2], b[3])
end = time.time()
print(end - start)

# plt.plot(np.abs(S))
# plt.plot(np.abs(S1))