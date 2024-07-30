# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:07:28 2023

@author: jlu
"""

import numpy as np
import scipy.signal
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
test_vertices=True
example_tethrahedral_neumann=False
example_tethrahedral_neumann_bond=False
example_Graph1MR=False
example_Graph1MR__=False
example_Junjie_Tw_method=False
simple_graph=False
#%%
from mesopylib.num.graphs.graph_bonds import Bond, Bond_absorbtion
from mesopylib.num.graphs.graph_vertices import Vertex \
    , Vertex_k, Vertex_circulator, Vertex_neumann, Vertex_dirichlet \
    , Vertex_open, Vertex_closed, Vertex_DFT, Vertex_1D_phase
from mesopylib.num.graphs.graph_num import Graph_VNA_effH, Graph_VNA \
    , create_h, get_min_eig, get_ev_h \
    , get_length, testConnectivity, set_vertex_chan_indices \
    , create_graph_fully_connected, create_graph_star, show_graph \
    , create_graph_scattering_matrix, create_graph_dd_matrix, create_bond_matrix \
    , create_connectivity_matrix \
    , create_S_matrix, create_S

    
#%% Junjie Tw_Methods_Compare 20230420
from junjie.num.graph.smatrix import effective_h, bond_scattering
from junjie.num.graph.examples import tetra_marburg_Wchange

k = np.linspace(0.01, 50, 20000)+1j*0.0001


vn_eff, vna_eff = tetra_marburg_Wchange(w=1)
S = vna_eff.calcS(k)
h = create_h(vn_eff,k)
ev_closed = get_ev_h(h,k)
plt.plot(k.real, np.abs(S))
print(ev_closed)

#%%

fn = 'D:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/2021-08-21-NonWeyl-Optimization/Tetrahedral_Graph/Simulation/length_change/'
bond_op  = np.load(fn+ 'bonds_length.npy',  allow_pickle=True, fix_imports=True)
ndim = 6
ndim1 = 8
l = np.zeros((ndim, ndim))
b = np.zeros((ndim1))

b = bond_op[:-1]

#l matrix (connecting bonds i,j with cable m l[i,j] = b[m])
l[0,1] = b[0]
l[0,3] = b[1]
l[0,4] = b[2]
l[1,2] = b[3]
l[1,4] = b[4]
l[2,3] = b[5]
l[2,4] = b[6]
l[3,5] = b[7]
l = l + l.T
nopen=1
wr = np.zeros((nopen, ndim))
wr[0,5] = 1
S,ei=effective_h(l, wr, k, True)
print(ei)
S00 = S[:,0]
plt.plot(k.real, np.abs(S00))
print(1-np.abs(np.mean(S00))**2)

