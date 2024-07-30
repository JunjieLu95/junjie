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
from junjie.num.graph.analytical import lasso
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
example_Bond_S=True
example_Heff_S=True

ndim1 = 3
b = np.zeros((ndim1))

b[0] = 4.631594805314004
b[1] = 0.3743445867654341
b[2] = 1.7514813406721745

phi = 3

lengths=b

k = np.linspace(0.01, 50, 2000)+1j*0.0001
if example_Bond_S:
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(2,name='v'+str(i+1)) for i in range(1)] 
    vn.append(Vertex_neumann(2, name='v2')) 
    vn.append(Vertex_neumann(3, name='v3')) 

    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[1])
    vn[1].connect2vertex(1,1,vn[2],bonds[2])
  
    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[2])
    testConnectivity(vn)
    # show_graph(vn)
    
    ss=create_graph_scattering_matrix(vn)
    # print(ss)
    dd=create_graph_dd_matrix(vn,k)
    sb=create_bond_matrix(vn,k)
    min_eig_close = np.min(np.abs(np.linalg.eig(np.eye(sb.shape[-1])-sb)[0]), axis=1)
    graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.4)[0]]
    S=create_S_matrix(sb,dd,vn, k, vna=vna, ports=[0])
    S1=create_S(vn, k, vna=vna)#, ports=[5])
    # plt.figure('S-bond');plt.clf()
    # plt.plot(k.real,np.abs(S[:,0,0]))
    plt.plot(k.real,np.abs(S1[:,0,0]))


#%%
from junjie.num.graph.smatrix import effective_h

ndim = 3
ndim1 = 3
l = np.zeros((ndim, ndim))
l[0,1] = b[0]
l[0,2] = b[1]
l[1,2] = b[2]
l = l + l.T

wr=[2]
    
wr = np.zeros((1, 3))
wr[0,2]=1
s3 = effective_h(l, wr, k, flag_eigenvalues=False)
plt.plot(k.real,np.abs(s3[:,0]), ':')
#%% Hans note

e2 = np.exp(1j*(k*(b[0]+b[1]+b[2])))

s4 = (3*e2-1)/(3-e2)  

plt.plot(k.real,np.abs(s4), ls=':')



