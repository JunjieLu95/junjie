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
    , Vertex_open, Vertex_closed, Vertex_DFT
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

ndim1 = 6
b = np.zeros((ndim1))

np.random.seed(156400)
b[0] = np.random.rand()
b[1] = np.random.rand()
b[2] = np.random.rand()
b[3] = np.random.rand()
b[4] = np.random.rand()
b[5] = np.random.rand()

#fix graphlength to 7 m
b = (b/b.sum()) * 7
lengths=b
k = np.linspace(0.01, 50, 20000)+1j*0.0001
if example_Bond_S:
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(4)] 
    vn[0]=Vertex_neumann(4,name='v1')
    # add vertex for dangling bond
    # vn.append(Vertex_neumann(1,name='v7'))
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,3,vn[0])
    testConnectivity(vn)
    # show_graph(vn)
    
    ss=create_graph_scattering_matrix(vn)
    dd=create_graph_dd_matrix(vn,k)
    sb=create_bond_matrix(vn,k)
    min_eig_close = np.min(np.abs(np.linalg.eig(np.eye(sb.shape[-1])-sb)[0]), axis=1)
    graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.4)[0]]
    S=create_S_matrix(sb,dd,vn, k, vna=vna, ports=[0])
    S1=create_S(vn, k, vna=vna)#, ports=[5])
    # plt.figure('S-bond');plt.clf()
    plt.plot(k.real,np.abs(S[:,0,0]))
    # plt.plot(k.real,np.abs(S1[:,0,0]))

if example_Heff_S:
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(4)] 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])

    vna_eff=Graph_VNA_effH(vn,connect=[0,0],couplings=[1,1])
    S=vna_eff.calcS(k)
    # plt.figure('S-bond');plt.clf()
    plt.plot(k.real,np.abs(S[:,0]))
    # plt.plot(k.real,np.abs(S1[:,0,0]))
    
from junjie.graph.num.smatrix import effective_h, bond_scattering
from junjie.phd_thesis.util_graph_examples import generate_thetra
l = generate_thetra(156400)
wr = np.zeros((1, 4))
wr[0,0]=1
s = effective_h(l, wr, k)
plt.plot(k.real,np.abs(s), '--')

wr = [0]
s2 = bond_scattering(l, wr, k)
plt.plot(k.real,np.abs(s2[:,0,0]), ':')


