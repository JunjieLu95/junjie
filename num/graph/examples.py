# -*- coding: utf-8 -*-
"""
This program is used for calculating the S-matrix of the open GOE Neumann graph.
For this purpose, first we need get the h maxtrix, then we can get the scattering matrix.
"""
import numpy as np
import matplotlib.pyplot as plt

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
# from caspar import sun_factorization, sun_reconstruction

def fully_connect_Weyl(seed, w, t_length = 7):
    ndim1 = 6
    # b = np.zeros((ndim1))
    lengths = generate_random_bonds_equal(n=ndim1, realizations=1, seed=seed, t_length=t_length)
    
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(4)] 
    # vn[0]=Vertex_neumann(3,name='v1')
    # add vertex for dangling bond
    # vn.append(Vertex_neumann(1,name='v7'))
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])

    vna=Graph_VNA_effH(vn,connect=[0],couplings=[w])
    
    return vn, vna
    

def tetra_marburg_Wchange(w):
    ndim1 = 8
    lengths = np.zeros((ndim1))
    
    lengths[0] = 0.94939222
    lengths[1] = 0.37434459
    lengths[2] = 1.75148134
    lengths[3] = 1.59136089
    lengths[4] = 0.86751784
    lengths[5] = 0.78565432
    lengths[6] = 0.43766953
    lengths[7] = 0.24352648
    
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(6)] 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[3],bonds[1])
    vn[0].connect2vertex(2,0,vn[4],bonds[2])
    vn[1].connect2vertex(1,0,vn[2],bonds[3])
    vn[1].connect2vertex(2,1,vn[4],bonds[4])
    vn[2].connect2vertex(1,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[4],bonds[6])
    vn[3].connect2vertex(2,0,vn[5],bonds[7])
    
    vna=Graph_VNA_effH(vn,connect=[5],couplings=[w])

    return vn, vna

def tetra_marburg(phi):
    ndim1 = 9
    lengths = np.zeros((ndim1))
    
    lengths[0] = 0.94939222
    lengths[1] = 0.37434459
    lengths[2] = 1.75148134
    lengths[3] = 1.59136089
    lengths[4] = 0.86751784
    lengths[5] = 0.78565432
    lengths[6] = 0.43766953
    lengths[7] = 0.24352648
    lengths[8] = 0
    
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(6)] 
    vn.append(Vertex_1D_phase(phi, name='v7')) 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[3],bonds[1])
    vn[0].connect2vertex(2,0,vn[4],bonds[2])
    vn[1].connect2vertex(1,0,vn[2],bonds[3])
    vn[1].connect2vertex(2,1,vn[4],bonds[4])
    vn[2].connect2vertex(1,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[4],bonds[6])
    vn[3].connect2vertex(2,0,vn[5],bonds[7])
    vn[6].connect2vertex(0,1,vn[5],bonds[8])
    
    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[5])

    return vn, vna

def lasso_graph(l_ring, l_d, phi):
    """
    In lasso_tetra_marburg, l_ring = 6.75742073, l_d = 0.24352648 (m)
    also with l_ring:l_d=1:1, 1:4, 2:3
    
    In lasso_Lanzhou, if there is no attunators,
    l_ring = l_d = 0.35262348 (m)
    
    """
    
    ndim1 = 5
    lengths = np.zeros((ndim1))
    
    lengths[0] = l_ring/3
    lengths[1] = l_ring/3
    lengths[2] = l_ring/3
    lengths[3] = l_d
    lengths[4] = 0

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(2,name='v'+str(i+1)) for i in range(1)] 
    vn.append(Vertex_neumann(3, name='v2')) 
    vn.append(Vertex_neumann(2, name='v3')) 
    vn.append(Vertex_neumann(3, name='v4')) 
    vn.append(Vertex_1D_phase(phi, name='v5')) 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[1])
    vn[1].connect2vertex(1,1,vn[2],bonds[2])
    vn[1].connect2vertex(2,0,vn[3],bonds[3])
    vn[3].connect2vertex(1,0,vn[4],bonds[4])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[3])

    return vn, vna

def Y_graph(l1, l2, l_d, phi):
    """
    Y graph, with all Neumann conditions in the vertexs
    
    """
    
    ndim1 = 4
    lengths = np.zeros((ndim1))
    
    lengths[0] = l1
    lengths[1] = l2
    lengths[2] = l_d
    lengths[3] = 0

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_open(name='v1')]
    vn.append(Vertex_open(name='v2')) 
    vn.append(Vertex_neumann(3, name='v3')) 
    vn.append(Vertex_neumann(3, name='v4')) 
    vn.append(Vertex_1D_phase(phi, name='v5')) 
    vn[0].connect2vertex(0,0,vn[2],bonds[0])
    vn[1].connect2vertex(0,1,vn[2],bonds[1])
    vn[2].connect2vertex(2,0,vn[3],bonds[2])
    vn[3].connect2vertex(1,0,vn[4],bonds[3])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[3])
        
    return vn, vna

def Thesis_WeylGraph_ex1():
    ndim1 = 6
    lengths = np.zeros((ndim1))
    
    lengths[0] = 1.22735567
    lengths[1] = 1.63394134
    lengths[2] = 1.42042372
    lengths[3] = 2.13124949
    lengths[4] = 0.36339327
    lengths[5] = 0.22363651
    
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(4)] 
    
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[4])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(1,1,vn[2],bonds[1])
    vn[1].connect2vertex(2,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[3],bonds[2])

    vna=Graph_VNA_effH(vn,connect=[0],couplings=[1])
    
    return vn, vna
    
    
def Thesis_NonWeylGraph_ex2():
    ndim1 = 8
    lengths = np.zeros((ndim1))
    
    lengths[0] = 0.94939222
    lengths[1] = 0.37434459
    lengths[2] = 1.75148134
    lengths[3] = 1.59136089
    lengths[4] = 0.86751784
    lengths[5] = 0.78565432
    lengths[6] = 0.43766953
    lengths[7] = 0.24352648
    
    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(6)] 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[3],bonds[1])
    vn[0].connect2vertex(2,0,vn[4],bonds[2])
    vn[1].connect2vertex(1,0,vn[2],bonds[3])
    vn[1].connect2vertex(2,1,vn[4],bonds[4])
    vn[2].connect2vertex(1,1,vn[3],bonds[5])
    vn[2].connect2vertex(2,2,vn[4],bonds[6])
    vn[3].connect2vertex(2,0,vn[5],bonds[7])
    
    vna=Graph_VNA_effH(vn,connect=[5],couplings=[1])
    
    return vn, vna

def triangle_graph_close(l1, l2, l3, la1, la2, la3):
    
    ndim1 = 6
    lengths = np.zeros((ndim1))
    
    lengths[0] = l1
    lengths[1] = l2
    lengths[2] = l3
    lengths[3] = la1
    lengths[4] = la2
    lengths[5] = la3

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(3)] 
    vn.append(Vertex_dirichlet(1, name='v4')) 
    vn.append(Vertex_dirichlet(1, name='v5')) 
    vn.append(Vertex_dirichlet(1, name='v6')) 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[1])
    vn[1].connect2vertex(1,1,vn[2],bonds[2])
    vn[0].connect2vertex(2,0,vn[3],bonds[3])
    vn[1].connect2vertex(2,0,vn[4],bonds[4])
    vn[2].connect2vertex(2,0,vn[5],bonds[5])
    
    return vn

def triangle_graph_open(l1, l2, l3, la1, la2, la3):
    
    ndim1 = 9
    lengths = np.zeros((ndim1))
    
    lengths[0] = l1
    lengths[1] = l2/2
    lengths[2] = l2/2
    lengths[3] = l3
    
    lengths[4] = 0.007
    lengths[5] = 0.007
    
    lengths[6] = la1
    lengths[7] = la2
    lengths[8] = la3

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(5)]
    vn.append(Vertex_dirichlet(1, name='v6')) 
    vn.append(Vertex_dirichlet(1, name='v7')) 
    vn.append(Vertex_dirichlet(1, name='v8')) 
    vn.append(Vertex_dirichlet(1, name='v9')) 
    
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[3])
    vn[1].connect2vertex(1,0,vn[3],bonds[2])
    vn[2].connect2vertex(1,1,vn[3],bonds[1])
    
    vn[3].connect2vertex(2,0,vn[4],bonds[4])
    vn[5].connect2vertex(0,1,vn[4],bonds[5])
    
    vn[0].connect2vertex(2,0,vn[6],bonds[6])
    vn[1].connect2vertex(2,0,vn[7],bonds[7])
    vn[2].connect2vertex(2,0,vn[8],bonds[8])
    
    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[4])
    
    return vn, vna

def triangle_graph_open_update(l1, l2, l3, la1, la2, la3):
    
    ndim1 = 15
    lengths = np.zeros((ndim1))
    
    lengths[0] = l1/2
    lengths[1] = l1/2
    lengths[2] = l2/2
    lengths[3] = l2/2
    lengths[4] = l3/2
    lengths[5] = l3/2
    
    lengths[6] = 0.007
    lengths[7] = 0.007
    
    lengths[8] = 0.007
    lengths[9] = 0.007
    
    lengths[10] = 0.007
    lengths[11] = 0.007
    
    lengths[12] = la1
    lengths[13] = la2
    lengths[14] = la3

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(3,name='v'+str(i+1)) for i in range(5)]
    vn.append(Vertex_dirichlet(1, name='v6')) 
    vn.append(Vertex_neumann(3, name='v6')) 
    vn.append(Vertex_neumann(3, name='v6')) 
    vn.append(Vertex_dirichlet(1, name='v6')) 
    vn.append(Vertex_neumann(3, name='v6')) 
    vn.append(Vertex_neumann(3, name='v6')) 
    vn.append(Vertex_dirichlet(1, name='v6')) 
    vn.append(Vertex_dirichlet(1, name='v7')) 
    vn.append(Vertex_dirichlet(1, name='v8')) 
    vn.append(Vertex_dirichlet(1, name='v9')) 
    
    vn[0].connect2vertex(0,0,vn[6],bonds[4])
    vn[0].connect2vertex(1,0,vn[9],bonds[0])
    vn[1].connect2vertex(0,1,vn[9],bonds[1])
    vn[1].connect2vertex(1,0,vn[3],bonds[2])
    vn[2].connect2vertex(0,1,vn[6],bonds[5])
    vn[2].connect2vertex(1,1,vn[3],bonds[3])
    
    vn[3].connect2vertex(2,0,vn[4],bonds[6])
    vn[5].connect2vertex(0,1,vn[4],bonds[7])
    
    vn[6].connect2vertex(2,0,vn[7],bonds[8])
    vn[8].connect2vertex(0,1,vn[7],bonds[9])
    
    vn[9].connect2vertex(2,0,vn[10],bonds[10])
    vn[11].connect2vertex(0,1,vn[10],bonds[11])
    
    vn[0].connect2vertex(2,0,vn[12],bonds[12])
    vn[1].connect2vertex(2,0,vn[13],bonds[13])
    vn[2].connect2vertex(2,0,vn[14],bonds[14])
    
    vna=Graph_VNA(n_ports=3, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[4])
    vna.connect2vertex(1,2,vn[7])
    vna.connect2vertex(2,2,vn[10])
    
    return vn, vna

def ring_graph_dp(l1, l2, la1, la2):
    
    ndim1 = 5
    lengths = np.zeros((ndim1))
    
    lengths[0] = l1/2
    lengths[1] = l1/2
    lengths[2] = l2
    lengths[3] = la1
    lengths[4] = la2

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(2,name='v'+str(i+1)) for i in range(1)] 
    vn.append(Vertex_neumann(3, name='v2')) 
    vn.append(Vertex_neumann(3, name='v3')) 
    vn.append(Vertex_neumann(1, name='v4')) 
    vn.append(Vertex_neumann(1, name='v5')) 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[0].connect2vertex(1,0,vn[2],bonds[1])
    vn[1].connect2vertex(1,1,vn[2],bonds[2])
    vn[1].connect2vertex(2,0,vn[3],bonds[3])
    vn[2].connect2vertex(2,0,vn[4],bonds[4])
    
    return vn

def chern_graph_not_work(phi1, phi2):
    
    bonds=[Bond(0.007,name='b'+str(i)) for i in range(4)]
    bond_phi1 = Bond(0.001,name='loop1', phase=[0,phi1])
    bond_phi2 = Bond(0.001,name='loop2', phase=[0,-phi2])
    vn=[Vertex_neumann(3,name='v'+str(i)) for i in range(4)] 
    vn.append(Vertex_dirichlet(1, name='v4')) 
    vn[0].connect2vertex(0,0,vn[4],bonds[0])
    vn[0].connect2vertex(1,0,vn[1],bonds[1])
    vn[1].connect2vertex(1,0,vn[2],bonds[2])
    vn[1].connect2vertex(2,0,vn[3],bonds[3])
    vn[2].connect2vertex(1,2,vn[2],bond_phi1)
    vn[3].connect2vertex(1,2,vn[3],bond_phi2)
    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[0])
    
    return vn, vna

def chern_graph(l1, l2, phi1, phi2):
    
    bonds=[Bond(0.007,name='b'+str(i)) for i in range(4)]
    bond_phi1 = Bond(l1, name='loop1', phase=[0,phi1])
    bond_phi1a = Bond(0,name='loop1a')
    bond_phi1b = Bond(0,name='loop1b')
    bond_phi2 = Bond(l2, name='loop2', phase=[phi2,0])
    bond_phi2a = Bond(0,name='loop2a')
    bond_phi2b = Bond(0,name='loop2b')
    vn=[Vertex_neumann(3,name='v'+str(i)) for i in range(4)] 
    vn.append(Vertex_dirichlet(1, name='v4')) 
    vn.append(Vertex_neumann(2,name='v5')) 
    vn.append(Vertex_neumann(2,name='v6'))
    vn.append(Vertex_neumann(2,name='v7'))
    vn.append(Vertex_neumann(2,name='v8'))
    
    vn[0].connect2vertex(0,0,vn[4],bonds[0])
    vn[0].connect2vertex(1,0,vn[1],bonds[1])
    vn[1].connect2vertex(1,0,vn[2],bonds[2])
    vn[1].connect2vertex(2,0,vn[3],bonds[3])
    
    vn[2].connect2vertex(1,0,vn[5],bond_phi1)
    vn[5].connect2vertex(1,0,vn[6],bond_phi1a)
    vn[6].connect2vertex(1,2,vn[2],bond_phi1b)
    
    vn[3].connect2vertex(1,0,vn[7],bond_phi2)
    vn[7].connect2vertex(1,0,vn[8],bond_phi2a)
    vn[8].connect2vertex(1,2,vn[3],bond_phi2b)
    
    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[0])
    
    return vn, vna

def is_unitary(m):
            return np.allclose(np.eye(m.shape[0]), m.H * m)
        
class Vertex_neumann_perturbation_2by2(Vertex):
    """
    Vertex with the variable 2*2 unitary matrix,
    S = ((a, b),(-e^(-i phi)b*, e^(-i phi)a*))
    when b=1, phi=pi, S = ((0, 1), (1, 0))
    
    Two parameters par_b and par_phi,
    and 0<par_b<1
    """
    def __init__(self, par_b=0, par_phi=0, name=None):
        b = 1 
        phi = np.pi
        b -= par_b
        phi += par_phi
        a = np.sqrt(1-b**2)
        S = np.array([[a,b],[-np.exp(1j*phi)*np.conj(b),np.exp(1j*phi)*np.conj(a)]])
        Vertex.__init__(self,S_in=S, name=name, vtype='Neumann')

class Vertex_neumann_perturbation_3by3(Vertex):
    """
    https://github.com/glassnotes/Caspar
    
    Vertex with the variable 3*3 unitary matrix
    Use the function "sun_factorization" for the s-matrix of T-junction
    S_T = 1/3*((-1, 2, 2), (2, -1, 2), (2, 2, -1))
    we can get three 2*2 SU(2) matrix, and then we can change the parameter
    
    """
    def __init__(self, par_pos=(0,0), par_value=0, name=None):
        fac_par = [('2,3', [0.0, np.pi/2, 0.0]),
                   ('1,2', [np.pi, 2*np.arccos(1/3), np.pi]),
                   ('2,3', [np.pi, np.pi/2, np.pi])]
        fac_par[par_pos[0]][1][par_pos[1]] += par_value
        S = sun_reconstruction(3, fac_par)
        Vertex.__init__(self,S_in=S, name=name, vtype='Neumann')
        
def lasso_graph_BIC(l_ring, l_d, phi, v22, v33):
    """
    In lasso_BIC, we introduce a vertex which can be varied to the loop bond
    
    """
    
    ndim1 = 6
    lengths = np.zeros((ndim1))
    
    lengths[0] = l_ring/4
    lengths[1] = l_ring/4
    lengths[2] = l_ring/4
    lengths[3] = l_ring/4
    lengths[4] = l_d
    lengths[5] = 0

    bonds=[Bond(ilength,name='b'+str(i)) for i,ilength in enumerate(lengths)]
    vn=[Vertex_neumann(2,name='v0')] 
    vn.append(v22) #v1
    vn.append(Vertex_neumann(2, name='v2'))
    vn.append(v33) #v3
    vn.append(Vertex_neumann(3, name='v4')) 
    vn.append(Vertex_1D_phase(phi, name='v5')) 
    vn[0].connect2vertex(0,0,vn[1],bonds[0])
    vn[1].connect2vertex(1,0,vn[2],bonds[1])
    vn[2].connect2vertex(1,0,vn[3],bonds[2])
    vn[3].connect2vertex(1,1,vn[0],bonds[3])
    vn[3].connect2vertex(2,0,vn[4],bonds[4])
    vn[4].connect2vertex(1,0,vn[5],bonds[5])

    vna=Graph_VNA(n_ports=1, couplings=[1], vertex_list=vn)
    vna.connect2vertex(0,2,vn[4])

    return vn, vna

# k = np.linspace(0.01, 50, 20000)+1j*0.0001
# vn, vna = tetra_marburg(phi=0.2)
# vn, vna = lasso_graph(l_ring=6.75742073, l_d=0.24352648, phi=0.2)
# S1 = create_S(vn, k, vna=vna)[:,0,0]
# plt.plot(k.real, np.abs(S1))

# vn_eff, vna_eff = tetra_marburg_Wchange(w=1)
# S = vna_eff.calcS(k)
# h = create_h(vn_eff,k)
# ev_closed = get_ev_h(h,k)
# plt.plot(k.real, np.abs(S))
# print(ev_closed)
