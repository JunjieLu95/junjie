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
from junjie.graph.num.analytical import Y_graph_phil2_w
import time
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
from mesopylib.num.graphs.graphs_dirichlet import generate_random_bonds_equal


def Y_graph_w_lphi(l1, l_d, phi_l2):
    """
    Y graph, with all Neumann conditions in the vertexs
    
    """
    
    ndim1 = 3
    lengths = np.zeros((ndim1))
    
    lengths[0] = l1
    lengths[1] = 0
    lengths[2] = l_d

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_open(name='v1')]
    vn.append(Vertex_1D_phase(phi, name='v2')) 
    vn.append(Vertex_neumann(3, name='v3')) 
    vn.append(Vertex_neumann(2, name='v4')) 
    
    vn[0].connect2vertex(0,0,vn[2],bonds[0])
    vn[1].connect2vertex(0,1,vn[2],bonds[1])
    vn[2].connect2vertex(2,0,vn[3],bonds[2])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,1,vn[3])
    
    return vn, vna
#%%
ndim1 = 2
b = np.zeros((ndim1))

b[0] = 0.6317
b[1] = 0.2230
phi = 7

lengths=b

k = np.linspace(0.01, 200, 20000)+1j*0.009


graph_info = (b[0], b[1], phi)
vn, vna = Y_graph_w_lphi(*graph_info)
S = create_S(vn, k, vna=vna)
# plt.plot(k.real,np.abs(s1[:,0,0]))
    
w=1

start = time.time()
S1 = Y_graph_phil2_w(k, b[0], phi, b[1], w)
end = time.time()
print(end - start)

plt.plot(np.abs(S[:,0,0]))
plt.plot(np.abs(S1))