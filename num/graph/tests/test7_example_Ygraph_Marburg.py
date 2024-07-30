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
from junjie.graph.num.examples import Y_graph
from junjie.graph.num.analytical import Y_graph_phic

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

def bond_scattering(l, wr, k, phi0):
    """ 
        'l' is the length matrix of the graph.
        'wr' is the open ports. 
         (wr=[4,5,6])
        'k' is the wave vector.
        'S' is the S-matrix of the graph.
    """
    ndim = len(l)
    nkmax = len(k)
    c = np.zeros((ndim, ndim))
    c[np.nonzero(l)]=1 
    c[3,4] = 1
    c[4,3] = 1
    ndim1 = int(np.sum(c)/2)
    
    # ports = [4,5,6]
    ports = wr                    # Define open lead in graph
    v = np.zeros(ndim)+1
    v[0]=1
    v[1]=1
    v[2]=3
    v[3]=3
    v[4]=1
    rho = np.zeros(ndim)+1
    for i in range(len(ports)):
        rho[ports[i]]=2/v[ports[i]]-1
    tau = np.zeros((ndim, ndim))
    for i in range(len(ports)):
        for j in range(ndim):
            tau[ports[i], j]=2/v[ports[i]]
    
    # dd = np.zeros((nkmax, 2*ndim1, 2*ndim1), dtype = 'complex_')  # Calculate the d matrix
    # for i in range(2*ndim1):
    #     dd[:,i,i]=np.exp(1j*k*l[np.nonzero(l)][i])
        
    dd = np.zeros((nkmax, (2*ndim1)**2), dtype = 'complex_')
    nn1=-1        # Calculate the d matrix if some bonds have open/short terminator
    for i1 in range(ndim): 
        for j1 in range(ndim):
            if c[i1,j1]!=0:
                nn1+=1
                phi=k*l[i1,j1]
                if i1==3 and j1==4:
                    phi=phi+phi0
                if i1==4 and j1==3:
                    phi=phi
                dd[:,nn1*(2*ndim1+1)]=np.exp(1j*phi)
    dd=np.reshape(dd, (nkmax, 2*ndim1, 2*ndim1))  
    
    nn1=-1
    ss = np.zeros((2*ndim1, 2*ndim1))
    for i1 in range(ndim):
        for j1 in range(ndim):
            if c[i1,j1]!=0:
                nn1+=1
                nn2=-1
                for i2 in range(ndim):
                    for j2 in range(ndim):
                        if c[i2,j2]!=0:
                            nn2+=1
                            if j1==i2:
                                ss[nn1,nn2]=2/v[j1]
                                if i1==j2:
                                    ss[nn1,nn2]=2/v[j1]-1
                                # if j1==3:         # Calculate the s matrix if graph have circulator or repalce the scattering matrix of some vertexs.
                                #     ss[nn1,nn2]=0
                                #     if i1==1 and j2==0:
                                #         ss[nn1,nn2]=1
                                #     if i1==0 and j2==2:
                                #         ss[nn1,nn2]=1
                                #     if i1==2 and j2==1:
                                #         ss[nn1,nn2]=1
                                if j1==4:
                                    ss[nn1,nn2]=0
                                    if i1==j2:
                                        ss[nn1,nn2]=1   #open ss=1; short ss=-1
                                if j1==0:
                                    ss[nn1,nn2]=0
                                    if i1==j2:
                                        ss[nn1,nn2]=1   #open ss=1; short ss=-1
                                if j1==1:
                                    ss[nn1,nn2]=0
                                    if i1==j2:
                                        ss[nn1,nn2]=1   #open ss=1; short ss=-1
    tt = ss+0
    print(ss)
    sb = np.einsum('nij,jk->nik', dd, tt)
    zinv = np.linalg.inv(np.eye(2*ndim1)-sb)  
    smat = np.zeros((nkmax,ndim,ndim), dtype = 'complex_')
    for ii in range(len(ports)):
        smat[:,ports[ii],ports[ii]]=rho[ports[ii]]
    nn1 = -1
    for i1 in range(ndim):
        for j1 in range(ndim):
            if c[i1,j1]!=0:
                nn1+=1
                nn2=-1
                for i2 in range(ndim):
                    for j2 in range(ndim):
                        if c[i2,j2]!=0:
                            nn2+=1
                            smat[:,i1,j2]=smat[:,i1,j2]+tau[i1,j1]*zinv[:,nn1,nn2]*dd[:,nn2,nn2]*tau[j2,i2]
    
    nonz = np.nonzero(smat)
    S = smat[nonz[0], nonz[1], nonz[2]]
    S = np.reshape(S, (nkmax, len(ports), len(ports)))
    # min_eig_close = np.min(np.abs(np.linalg.eig(np.eye(2*ndim1)-sb)[0]), axis=1)
    # graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.01)[0]]
    # poles_open = np.min(np.abs(np.linalg.eig(np.eye(len(ports))-S)[0]), axis=1)
    # graph_poles_open = k[scipy.signal.find_peaks(-poles_open, height=-0.01)[0]]
    return S, ss, dd
    
#%% Junjie Tw_Methods_Compare 20230420
example_Bond_S=True
example_Heff_S=True

ndim1 = 4
b = np.zeros((ndim1))

b[0] = 4.631594805314004
b[1] = 0.3743445867654341
b[2] = 0.24352647777101874
b[3] = 0
phi = 6

lengths=b

k = np.linspace(0.01, 20, 20000)+1j*0.009
if example_Bond_S:
    graph_info = (b[0], b[1], b[2], phi)
    vn, vna = Y_graph(*graph_info)
    s1 = create_S(vn, k, vna=vna)
    plt.plot(k.real,np.abs(s1[:,0,0]))
    
#%%
# a1 = np.cos(k*b[0])
# a2 = np.cos(k*b[1])
# a3 = np.sin(k*(b[0]+b[1]))
# ed = np.exp(1j*(k*b[2]))**2
# G1 = -np.tan(phi/2)
# G2 = (1j*a3*(1+ed)+a1*a2*(-1+ed))/(-1j*a1*a2*(1+ed)+a3*(-1+ed))

# # s3 = (1-1j*G2)/(1+1j*G2)
# s4 = -(1-1j*G1-1j*G2)/(1+1j*G1+1j*G2)
s4 = Y_graph_phic(k, b[0], b[1], b[2], phi)
# s44 = Y_graph_smatrix_approach2(k, b[0], b[1], b[2], phi)

plt.plot(k.real,np.abs(s4), ls=':')
# plt.plot(k.real,np.abs(s44), ls=':')

#%%
ndim = 5
ndim1 = 4
l = np.zeros((ndim, ndim))
l[0,2] = b[0]
l[1,2] = b[1]
l[2,3] = b[2]
l[3,4] = b[3]
l = l + l.T

wr=[3]
    
s2, ss2, dd2 = bond_scattering(l, wr, k, phi)
plt.plot(k.real,np.abs(s2[:,0,0]), '-.')

#%%
from junjie.graph.num.smatrix import effective_h
ndim = 4
l = np.zeros((ndim, ndim))
l[0,2] = b[0]
l[1,2] = b[1]
l[2,3] = b[2]
l = l + l.T

wr = np.zeros((1, 4))
wr[0,3]=1
s3 = effective_h(l, wr, k, flag_eigenvalues=False)
plt.plot(k.real,np.abs(s3), ls=':')

#%%
ndim = 3
ndim1 = 2
l = np.zeros((ndim, ndim))
l[0,1] = b[0]
l[0,2] = b[1]
l = l + l.T

wr = np.zeros((1, 3))
wr[0,0]=1
s5 = effective_h(l, wr, k, flag_eigenvalues=False)
plt.plot(k.real,np.abs(s5[:,0]), '-.')

a1 = np.cos(k*b[0])
a2 = np.cos(k*b[1])
a3 = np.sin(k*(b[0]+b[1]))
g = a1*a2/a3
s6 = (1-1j*g)/(1+1j*g)
plt.plot(k.real,np.abs(s6[:]), '-.')

