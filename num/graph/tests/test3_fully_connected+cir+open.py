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
    c[1,4]=1
    c[4,1]=1
    ndim1 = int(np.sum(c)/2)
    
    # ports = [4,5,6]
    ports = wr                    # Define open lead in graph
    v = np.zeros(ndim)+3             # Define the valancy for every vertex
    
    v[0] = 4
    v[1] = 4
    v[4] = 1
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
                if i1==4 and j1==1:
                    phi=phi+phi0
                if i1==1 and j1==4:
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
                                if j1==3:         # Calculate the s matrix if graph have circulator or repalce the scattering matrix of some vertexs.
                                    ss[nn1,nn2]=0
                                    if i1==1 and j2==0:
                                        ss[nn1,nn2]=1
                                    if i1==0 and j2==2:
                                        ss[nn1,nn2]=1
                                    if i1==2 and j2==1:
                                        ss[nn1,nn2]=1
                                if j1==4:
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

ndim1 = 7
b = np.zeros((ndim1))

np.random.seed(156400)
b[0] = np.random.rand()
b[1] = np.random.rand()
b[2] = np.random.rand()
b[3] = np.random.rand()
b[4] = np.random.rand()
b[5] = np.random.rand()
b[6] = 0

#fix graphlength to 7 m
b = (b/b.sum()) * 7
lengths=b

phi=np.pi

k = np.linspace(0.01, 50, 20000)+1j*0.0001
if example_Bond_S:
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(4)] 
    vn[0]=Vertex_neumann(4,name='v1')
    vn[1]=Vertex_neumann(4,name='v2')
    vn[3]=Vertex_circulator(3)
    # vn.append(Vertex_open( name='v5')) 
    vn.append(Vertex_1D_phase(phi, name='v5')) 
    # add vertex for dangling bond
    # vn.append(Vertex_neumann(1,name='v7'))
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])
    vn[1].connect2vertex(3,0,vn[4],bonds[6])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,3,vn[0])
    testConnectivity(vn)
    # show_graph(vn)
    
    ss=create_graph_scattering_matrix(vn)
    print(ss)
    dd=create_graph_dd_matrix(vn,k)
    sb=create_bond_matrix(vn,k)
    min_eig_close = np.min(np.abs(np.linalg.eig(np.eye(sb.shape[-1])-sb)[0]), axis=1)
    graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.4)[0]]
    S=create_S_matrix(sb,dd,vn, k, vna=vna, ports=[0])
    S1=create_S(vn, k, vna=vna)#, ports=[5])
    # plt.figure('S-bond');plt.clf()
    plt.plot(k.real,np.abs(S[:,0,0]))
    plt.plot(k.real,np.abs(S1[:,0,0]))

    
from junjie.graph.num.smatrix import effective_h
from junjie.phd_thesis.util_graph_examples import generate_thetra
l = np.zeros((5,5))
l[:4,:4]= generate_thetra(156400)
wr = np.zeros((1, 5))

wr = [0]
s2, ss2, dd2 = bond_scattering(l, wr, k, phi)
plt.plot(k.real,np.abs(s2[:,0,0]), ':')
# plt.plot(k.real,np.imag(s2[:,0,0]), ':')

