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

ndim1 = 14
b = np.zeros((ndim1))

np.random.seed(156400)
b[0] = np.random.rand()
b[1] = np.random.rand()
b[2] = np.random.rand()
b[3] = np.random.rand()
b[4] = np.random.rand()
b[5] = np.random.rand()
for i in range(6):
    b[i+6] = b[i]
b[-1] = 0.7
b[-2] = 0.7

#fix graphlength to 7 m
# b = (b/b.sum()) * 7
lengths=b
k = np.linspace(0.01, 50, 2000)+1j*0.0001
if example_Bond_S:
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(9)] 
    vn[0]=Vertex_neumann(4,name='v1')
    # add vertex for dangling bond
    # vn.append(Vertex_neumann(1,name='v7'))
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])
    
    vn[4]=Vertex_neumann(4,name='v4')
    # add vertex for dangling bond
    # vn.append(Vertex_neumann(1,name='v7'))
    vn[4].connect2vertex(0,0,vn[5],bonds[6])
    vn[4].connect2vertex(1,0,vn[6],bonds[10])
    vn[4].connect2vertex(2,0,vn[7],bonds[9])
    vn[5].connect2vertex(1,1,vn[6],bonds[7])
    vn[5].connect2vertex(2,1,vn[7],bonds[11])
    vn[6].connect2vertex(2,2,vn[7],bonds[8])
    
    vn[8].connect2vertex(0,3,vn[0],bonds[-1])
    vn[8].connect2vertex(1,3,vn[4],bonds[-2])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[8])
    testConnectivity(vn)
    # show_graph(vn)
    
    # ss=create_graph_scattering_matrix(vn)
    # dd=create_graph_dd_matrix(vn,k)
    # sb=create_bond_matrix(vn,k)
    # min_eig_close = np.min(np.abs(np.linalg.eig(np.eye(sb.shape[-1])-sb)[0]), axis=1)
    # graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.4)[0]]
    # S=create_S_matrix(sb,dd,vn, k, vna=vna, ports=[0])
    S1=create_S(vn, k, vna=vna)#, ports=[5])
    # plt.figure('S-bond');plt.clf()
    # plt.plot(k.real,np.abs(S[:,0,0]))
    plt.plot(k.real,np.abs(S1[:,0,0]))
    S1 = S1[:,0,0]
#%%

b = b[:7]
b[-1] = 0.7

lengths=b
k = np.linspace(0.01, 50, 2000)+1j*0.0001
if example_Bond_S:
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(4)] 
    vn[0]=Vertex_neumann(4,name='v1')
    # add vertex for dangling bond
    vn.append(Vertex_neumann(2,name='v7'))
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])
    vn[0].connect2vertex(3,0,vn[4],bonds[-1])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,1,vn[4])
    testConnectivity(vn)
    # show_graph(vn)
    
    # ss=create_graph_scattering_matrix(vn)
    # dd=create_graph_dd_matrix(vn,k)
    # sb=create_bond_matrix(vn,k)
    # min_eig_close = np.min(np.abs(np.linalg.eig(np.eye(sb.shape[-1])-sb)[0]), axis=1)
    # graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.4)[0]]
    # S=create_S_matrix(sb,dd,vn, k, vna=vna, ports=[0])
    S2=create_S(vn, k, vna=vna)#, ports=[5])
    S2 = S2[:,0,0]
    # plt.figure('S-bond');plt.clf()
    # plt.plot(k.real,np.abs(S[:,0,0]))
    # plt.plot(k.real,np.abs(S1[:,0,0]))
#%%
G2 = -1j*(1-S2)/(1+S2)
SS = -(1-2j*G2)/(1+2j*G2)

# G2 = -1j*(1+S2)/(1-S2) # Junjie
# SS = (G2**2+2j*G2)/(G2**2-2j*G2)

plt.plot(k.real,np.abs(S1))
plt.plot(k.real,np.abs(SS))

#%%
def Y_graph(k, l1, l2, ld, lphi_c):
    a1 = np.cos(k*l1)
    a2 = np.cos(k*l2)
    a3 = np.sin(k*(l1+l2))
    ed = np.exp(2j*(k*ld))
    S1 = np.exp(2j*(k*lphi_c))
    G1 = -1j*(1-S1)/(1+S1)
    G2 = (a1*a2*(-1+ed)+1j*a3*(1+ed))/(a3*(-1+ed)-1j*a1*a2*(1+ed))
    S = -(1-1j*G1-1j*G2)/(1+1j*G1+1j*G2)
    return -S

def from_G_to_S(G):
    S = -(1-1j*G)/(1+1j*G)
    return S
    
def from_S_to_G(S):
    G = -1j*(1+S)/(1-S)
    return G

def T_jun(G1, G2):
    S = (G1*G2+1j*(G1+G2))/(G1*G2-1j*(G1+G2))
    return S

def Y_graph_Junjie(k, l1, l2, ld, lphi_c):
    G1 = from_S_to_G(np.exp(2j*(k*l1)))
    G2 = from_S_to_G(np.exp(2j*(k*l2)))
    S12 = T_jun(G1, G2)
    S12d = S12*np.exp(2j*(k*ld))
    G12d = from_S_to_G(S12d)
    G3 = from_S_to_G(np.exp(2j*(k*lphi_c)))
    return T_jun(G12d, G3)

def Y_graph_two_phi(k, l1, phi_l2, ld, phi_c):
    e1 = np.exp(2j*(k*l1))
    Gl2 = -np.tan(phi_l2/2)
    e2 = (1-1j*Gl2)/(1+1j*Gl2)
    ed = np.exp(2j*(k*ld))
    G2 = 1j*(-3-e1-e2+e1*e2-ed+e1*ed+e2*ed+3*e1*e2*ed)/(3+e1+e2-e1*e2-ed+e1*ed+e2*ed+3*e1*e2*ed)
    G1 = -np.tan(phi_c/2)
    S = -(1-1j*G1-1j*G2)/(1+1j*G1+1j*G2)
    return -S

def Y_graph_two_phi_Junjie(k, l1, phi_l2, ld, phi_c):
    G1 = from_S_to_G(np.exp(2j*(k*l1)))
    G2 = from_S_to_G(np.exp(2j*(phi_l2/2)))
    S12 = T_jun(G1, G2)
    S12d = S12*np.exp(2j*(k*ld))
    G12d = from_S_to_G(S12d)
    G3 = from_S_to_G(np.exp(2j*(phi_c/2)))
    return T_jun(G12d, G3)


k = np.linspace(0.01, 50, 2000)+1j*0.0001
# S11 = Y_graph(k, 1, 1.7, 0.7, 2.7)
# S22 = Y_graph_Junjie(k, 1, 1.7, 0.7, 2.7)
    
# plt.plot(k.real,np.abs(S11))
# plt.plot(k.real,np.abs(S22))
    
S11 = Y_graph_two_phi(k, 1, 1.7, 0.7, 2.7)
S22 = Y_graph_two_phi_Junjie(k, 1, 1.7, 0.7, 2.7)
    
plt.plot(k.real,np.abs(S11))
plt.plot(k.real,np.abs(S22))
    
    
    

