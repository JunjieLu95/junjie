# -*- coding: utf-8 -*-
"""
This program is used for calculating the S-matrix of the open GOE Neumann graph.
For this purpose, first we need get the h maxtrix, then we can get the scattering matrix.
"""
import numpy as np
import scipy.signal
# from scipy.constants import c as speed_light

def effective_h(l, wr, k, flag_eigenvalues=False):
    """ 
        'l' is the length matrix of the graph.
        'wr' is the leads–vertices coupling matrix, wr=1 when M=V. 
         (wr = np.zeros((nopen, ndim)) wr[0,4] = 1 wr[1,5] = 1 wr[2,6] = 1)
        'k' is the wave vector.
        'f' is the frequency.
        'S' is the S-matrix of the graph.
    """
    ndim = len(l)
    nkmax = len(k)
    nopen = len(wr)
    c = np.zeros((ndim, ndim))
    h = np.zeros((nkmax, ndim, ndim), dtype = 'complex_')
    
    c[np.nonzero(l)]=1                      # Connectivity matrix of the graph
  
    for i in range(ndim):
        nz = np.nonzero(l[i,:])[0]          # Find the position of nonzero element
        nz_l = l[i,nz]                      # Get the nonzero elements of l matrix
        nz_c = c[i,nz]                      # Get the nonzero elements of c matrix
        for j in range(len(nz)):
            h[:,i,i] = h[:,i,i]-nz_c[j]*np.cos(k*nz_l[j])/np.sin(k*nz_l[j])     # Calculate the diagonal elements of h matrix
            h[:,i,nz[j]] = nz_c[j]/np.sin(k*nz_l[j])        # Calculate the non-diagonal elements of h matrix
    
    wr1 = np.zeros((nkmax, nopen, ndim)) + wr
    wr2 = np.zeros((nkmax, ndim, nopen)) + wr.T
    wr3 = np.einsum('nij,njk->nik', wr2, wr1)           
    zinv = np.linalg.inv(h+wr3*1j)                  # (h+iW^T * W)^{-1}
    wr4 = np.einsum('nij,njk->nik', wr1, zinv)     
    wr5 = np.einsum('nij,njk->nik', wr4, wr2)       # W * zinv * W^T
    smat = np.zeros((nkmax, nopen, nopen)) - np.eye(nopen) + wr5*2j     # S = 2iW * (h + iW^T * W)^{-1} * W^T -1
    S = np.reshape(smat, (nkmax, nopen**2))
    
    if flag_eigenvalues:
        min_eig_close = np.min(np.abs(np.linalg.eig(h)[0]), axis=1)
        graph_eig_close = k[scipy.signal.find_peaks(-min_eig_close, height=-0.3)[0]]
        return S, graph_eig_close
    else:  
        return S

def bond_scattering(l, wr, k):
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
    ndim1 = int(np.sum(c)/2)
    
    # ports = [4,5,6]
    ports = wr                    # Define open lead in graph
    v = np.zeros(ndim)+3             # Define the valancy for every vertex
    
    v[0] = 4
    rho = np.zeros(ndim)+1
    for i in range(len(ports)):
        rho[ports[i]]=2/v[ports[i]]-1
    tau = np.zeros((ndim, ndim))
    for i in range(len(ports)):
        for j in range(ndim):
            tau[ports[i], j]=2/v[ports[i]]
    
    dd = np.zeros((nkmax, 2*ndim1, 2*ndim1), dtype = 'complex_')  # Calculate the d matrix
    for i in range(2*ndim1):
        dd[:,i,i]=np.exp(1j*k*l[np.nonzero(l)][i])
        
    # dd = np.zeros((nkmax, (2*ndim1)**2), dtype = 'complex_')
    # nn1=-1        # Calculate the d matrix if some bonds have open/short terminator
    # for i1 in range(ndim): 
    #     for j1 in range(ndim):
    #         if c[i1,j1]!=0:
    #             nn1+=1
    #             phi=k*l[i1,j1]
    #             if short==True:         
    #                 if i1==1 and j1==0:
    #                     phi=phi+np.pi
    #             dd[:,nn1*(2*ndim1+1)]=np.exp(1j*phi)
    # dd=np.reshape(dd, (nkmax, 2*ndim1, 2*ndim1))  
    
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
                                # if j1==0:         # Calculate the s matrix if graph have circulator or repalce the scattering matrix of some vertexs.
                                #     ss[nn1,nn2]=0
                                #     if i1==1 and j2==5:
                                #         ss[nn1,nn2]=1
                                #     if i1==5 and j2==6:
                                #         ss[nn1,nn2]=1
                                #     if i1==6 and j2==1:
                                #         ss[nn1,nn2]=1
    tt = ss+0
    
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
    return S

