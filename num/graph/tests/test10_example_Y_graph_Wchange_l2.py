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
from junjie.graph.num.analytical import Y_graph_w
import time

start = time.time()

w=9
b = np.zeros((3))
b[0] = 0.6317
b[1] = 0.3183
b[2] = 0.2230
    
ndim = 4
l = np.zeros((ndim, ndim))
l[0,2] = b[0]
l[1,2] = b[1]
l[2,3] = b[2]
l = l + l.T
nopen=1
wr = np.zeros((nopen, ndim))
wr[0,3] = w
frq = np.linspace(0.1, 18, 64001)
k=frq2k(frq)*1e9
kr = k.real

k=kr+1j*0.09
S = effective_h(l, wr, k, flag_eigenvalues=False)

end = time.time()
print(end - start)
    
start = time.time()
S1 = Y_graph_w(k, b[0], b[1], b[2], w)
end = time.time()
print(end - start)

plt.plot(np.abs(S))
plt.plot(np.abs(S1))