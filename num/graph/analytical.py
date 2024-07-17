# -*- coding: utf-8 -*-
"""
Note: in the analytical drive we use microwave technology, 
psi = a*exp(-ikx)-b*exp(-ikx)
PRE 104, 045211 (2021)

and for the graph it has a different defination
psi = a*exp(-ikx)+b*exp(ikx)   
Ann. Phys. (NY) 274, 76 (1999)

So in all the analytical drive function,
the return S value always has a '-' sign

"""
import numpy as np 

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

def Y_graph_phic(k, l1, l2, ld, phi_c):
    a1 = np.cos(k*l1)
    a2 = np.cos(k*l2)
    a3 = np.sin(k*(l1+l2))
    ed = np.exp(2j*(k*ld))
    G1 = -np.tan(phi_c/2)
    G2 = (a1*a2*(-1+ed)+1j*a3*(1+ed))/(a3*(-1+ed)-1j*a1*a2*(1+ed))
    S = -(1-1j*G1-1j*G2)/(1+1j*G1+1j*G2)
    return -S 

def Y_graph_two_phi(k, l1, phi_l2, ld, phi_c):
    e1 = np.exp(2j*(k*l1))
    Gl2 = -np.tan(phi_l2/2)
    e2 = (1-1j*Gl2)/(1+1j*Gl2)
    ed = np.exp(2j*(k*ld))
    G2 = 1j*(-3-e1-e2+e1*e2-ed+e1*ed+e2*ed+3*e1*e2*ed)/(3+e1+e2-e1*e2-ed+e1*ed+e2*ed+3*e1*e2*ed)
    G1 = -np.tan(phi_c/2)
    S = -(1-1j*G1-1j*G2)/(1+1j*G1+1j*G2)
    return -S

def Y_graph_two_phi_ratio2(phi_l2, phi_c):
    G_lphi = -np.tan(phi_c/2)
    G_l2 = -np.tan(phi_l2/2)
    return -16*(8*G_lphi*(2+G_l2*1j)*G_l2**3 
                +4*G_l2**3*(-2j+G_l2) 
                +G_lphi**4*(27+9*G_l2**2+G_l2**4) 
                -4j*G_lphi**3*(27+9*G_l2**2-1j*G_l2**3+G_l2**4)
                -4*G_lphi**2*(27+9*G_l2**2-3j*G_l2**3+2*G_l2**4))

def Y_graph_w(k, l1, l2, ld, w):
    a1 = np.cos(k*l1)
    a2 = np.cos(k*l2)
    a3 = np.sin(k*(l1+l2))
    ed = -np.exp(2j*(k*ld))
    G2 = (a1*a2*(-1+ed)+1j*a3*(1+ed))/(a3*(-1+ed)-1j*a1*a2*(1+ed))
    S = (1-1j*G2*(w**2))/(1+1j*G2*(w**2))
    return -S

def Y_graph_phil2_w(k, l1, phi_l2, ld, w):
    e1 = np.exp(2j*(k*l1))
    Gl2 = -np.tan(phi_l2/2)
    e2 = (1-1j*Gl2)/(1+1j*Gl2)
    ed = -np.exp(2j*(k*ld))
    G2 = 1j*(-3-e1-e2+e1*e2-ed+e1*ed+e2*ed+3*e1*e2*ed)/(3+e1+e2-e1*e2-ed+e1*ed+e2*ed+3*e1*e2*ed)
    S = (1-1j*G2*(w**2))/(1+1j*G2*(w**2))
    return -S 

def lasso(k, lc, ld):
    ec = np.exp(1j*(k*lc))
    ed = np.exp(1j*(k*ld))
    S = ed**2*(3*ec-1)/(3-ec)
    return S

def lasso_phic(k, lc, ld, phi_c):
    ec = np.exp(1j*(k*lc))
    ed = np.exp(1j*(k*ld))
    G1 = -np.tan(phi_c/2)
    S2 = ed**2*(3*ec-1)/(3-ec)
    G2 = -1j*(1-S2)/(1+S2)
    S = -(1-1j*G1-1j*G2)/(1+1j*G1+1j*G2)
    return -S

def transform_S_to_fit(S):
    """
    This transformation will solve the problem that S parameter cannot be fitted
    """
    return -np.conj(S)

def Y_graph_2couple_s12(k, l1, phi_l2, ld, phi_c):
    f1=np.tan(phi_c/2)
    f2=1/np.tan(k*ld)
    f3=np.tan(k*l1)
    f4=np.tan(phi_l2/2)
    g=1/np.sin(k*ld)
    h = np.zeros((len(k),2,2), complex)
    h[:,0,0]=f1-f2
    h[:,0,1]=g
    h[:,1,0]=g
    h[:,1,1]=-f2+f3+f4
    
    nkmax = len(k)
    nopen = 2
    ndim = 2
    wr = np.zeros((2,2))
    wr[0,0]=1
    wr[1,1]=0.01
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    S = np.reshape(smat, (nkmax, nopen**2))
    
    return S[:,1]

def Y_graph_2couple(k, l1, phi_l2, ld, phi_c):
    f1=np.tan(phi_c/2)
    f2=1/np.tan(k*ld)
    f3=np.tan(k*l1)
    f4=np.tan(phi_l2/2)
    g=1/np.sin(k*ld)
    h = np.zeros((1,2,2), complex)
    h[:,0,0]=f1-f2
    h[:,0,1]=g
    h[:,1,0]=g
    h[:,1,1]=-f2+f3+f4
    
    nkmax = 1
    nopen = 2
    ndim = 2
    wr = np.zeros((2,2))
    wr[0,0]=1
    wr[1,1]=0.01
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    # S = np.reshape(smat, (nkmax, nopen**2))
    
    return smat

def Y_graph_2couple_l34_s12(k, l1, phi_l2, ld, phi_c, phi_l34):
    f1=np.tan(phi_c/2)
    f2=1/np.tan(k*ld)
    f3=np.tan(k*l1)
    f4=np.tan(phi_l2/2)
    g=1/np.sin(k*ld)
    h = np.zeros((len(k),2,2), complex)
    h[:,0,0]=f1-f2
    h[:,0,1]=g
    h[:,1,0]=g*np.exp(1j*phi_l34)
    h[:,1,1]=-f2+f3+f4
    
    nkmax = len(k)
    nopen = 2
    ndim = 2
    wr = np.zeros((2,2))
    wr[0,0]=1
    wr[1,1]=0.01
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    S = np.reshape(smat, (nkmax, nopen**2))
    
    return S[:,1]

def Y_graph_2couple_l34(k, l1, phi_l2, ld, phi_c, phi_l34):
    f1=np.tan(phi_c/2)
    f2=1/np.tan(k*ld)
    f3=np.tan(k*l1)
    f4=np.tan(phi_l2/2)
    g=1/np.sin(k*ld)
    h = np.zeros((len(k),2,2), complex)
    h[:,0,0]=f1-f2
    h[:,0,1]=g
    h[:,1,0]=g*np.exp(1j*phi_l34)
    h[:,1,1]=-f2+f3+f4
    
    nkmax = len(k)
    nopen = 2
    ndim = 2
    wr = np.zeros((2,2))
    wr[0,0]=1
    wr[1,1]=0.01
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    # S = np.reshape(smat, (nkmax, nopen**2))
    
    return smat

def Y_graph_2couple_s12_larger_W(k, l1, phi_l2, ld, phi_c):
    f1=np.tan(phi_c/2)
    f2=1/np.tan(k*ld)
    f3=np.tan(k*l1)
    f4=np.tan(phi_l2/2)
    g=1/np.sin(k*ld)
    h = np.zeros((len(k),2,2), complex)
    h[:,0,0]=f1-f2
    h[:,0,1]=g
    h[:,1,0]=g
    h[:,1,1]=-f2+f3+f4
    
    nkmax = len(k)
    nopen = 2
    ndim = 2
    wr = np.zeros((2,2))
    wr[0,0]=1
    wr[1,1]=1
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    S = np.reshape(smat, (nkmax, nopen**2))
    
    return S[:,1]

def Y_graph_2couple_s11_larger_W(k, l1, phi_l2, ld, phi_c):
    f1=np.tan(phi_c/2)
    f2=1/np.tan(k*ld)
    f3=np.tan(k*l1)
    f4=np.tan(phi_l2/2)
    g=1/np.sin(k*ld)
    h = np.zeros((len(k),2,2), complex)
    h[:,0,0]=f1-f2
    h[:,0,1]=g
    h[:,1,0]=g
    h[:,1,1]=-f2+f3+f4
    
    nkmax = len(k)
    nopen = 2
    ndim = 2
    wr = np.zeros((2,2))
    wr[0,0]=1
    wr[1,1]=1
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    S = np.reshape(smat, (nkmax, nopen**2))
    
    return S[:,0]

def Y_graph_s11_l34_loss(k, phi, lambda_l34):
    f1=1/np.tan(phi)
    f2=1/np.tan(phi+1j*lambda_l34)
    f3=1/np.tan(phi)
    f4=1/np.tan(phi)
    g=1/np.sin(phi+1j*lambda_l34)
    h = np.zeros((len(k),2,2), complex)
    h[:,0,0]=-f1-f2
    h[:,0,1]=g
    h[:,1,0]=g
    h[:,1,1]=-f2-f3-f4
    
    nkmax = len(k)
    nopen = 1
    ndim = 1
    wr = np.zeros((1,1))
    wr[0,0]=1
    # wr[1,1]=0.01
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    S = np.reshape(smat, (nkmax, nopen**2))
    
    return S[:,0]