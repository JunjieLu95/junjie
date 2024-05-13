# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:25:54 2024

@author: jlu
"""
import random
import numpy as np
import mesopylib.num.rmt.rmt_num as r
from junjie.tools import get_lim

def ei_dynamics_super(M, N, kappa, n_lambda, Im_range, E_range=None): 
    '''
    
    This code is designed to generate dynamics of zero/poles,
    with specific starting point.

    Parameters
    ----------
    M, N, kappa : int/float
        the parameters to construct H_eff.
    n_lambda : int
        the number of the realzations you want to have.
    Im_range : tuple
        the range of the starting point pole/zero
        (if one of the eigenvalues of the H_eff[0], its imaginary part
         is in the range of Im_range, then the code start to create the 
         dynamics, for example Im_range = (-21.63, -16.63), note the sign 
         should be negetive).
    E_range : tuple
        The default is (-N*0.01, N*0.01).

    Returns
    -------
    infos : tuple
        (fixHrand1, fixHrand2, M, N, kappa, lamda).
    dy_zero : array
        array of poles

    '''
    super_states = True
    while super_states:
        fixHrand1 = random.randint(1,900000000)
        fixHrand2 = random.randint(1,900000000)
        H0 = r.createRmtHWithSeed(fixHrand1, N, beta=1)
        H1 = r.createRmtHWithSeed(fixHrand2, N, beta=1)
        V = r.osys.channels.create_channels_with_given_coupling(
                    M, N, kappa, random_container=False)
        
        index_lamba = random.randint(0,1500-n_lambda-50)
        lamda = np.linspace(0,np.pi,1500)[index_lamba:index_lamba+n_lambda]

        heff = r.set_up_H_eff(H0 * np.cos(lamda[0]) + H1 * np.sin(lamda[0]), V)
        ei_heff = np.linalg.eigvals(heff)
        
        if E_range is None: 
            ev_lim_real = get_lim(ei_heff, (-N*0.01, N*0.01), 'real')
        else:
            ev_lim_real = get_lim(ei_heff, (E_range[0], E_range[1]), 'real')

        ev_lim_real_imag = get_lim(ev_lim_real, (Im_range[0], Im_range[1]), 'imag')
        
        if len(ev_lim_real_imag)==1:
            super_states = False
        else:
            super_states = True
    
    dy_zero = np.zeros((len(lamda)), dtype=complex)
    dy_zero[0] = ev_lim_real_imag
    for i in range(1,len(lamda)):
        heff = r.set_up_H_eff(H0 * np.cos(lamda[i]) + H1 * np.sin(lamda[i]), V)
        ei_heff = np.linalg.eigvals(heff)
        dy_zero[i] = ei_heff[np.argmin(np.abs(ei_heff-dy_zero[i-1]))]
    
    infos = (fixHrand1, fixHrand2, M, N, kappa, lamda)
    return dy_zero, infos

def velocity_collect(ev_connected_all, rescale=False, Erange=[-1,1], 
                     cutRange=[0, 1], flag_exp=False):
    """
    This function can give the velocity of those eigenvalues with small and 
    large imaginary part.

    Parameters
    ----------
    ev_connected_all : array of object
    
        ev_connected_all is connected eigenvalues matrix, 
        .shape[0] is number of dynamics from lambda (n_pars)
        .shape[1] is number of realizations (n_Evs)
        
        # ev_connected_all[1] is infos
    rescale : Ture or False
        if it is True, the velocity will be rescaled
    Erange : list
        the cut for Energy(real part of eigenvalues), range from Erange[0] to Erange[1]
    cutRange : list
        the cut for large imaginary part of eigenvalues, 
        first range is from cutRange_small[0] to cutRange_small[1]
        second range is from cutRange_super[0] to cutRange_super[1]
    flag_exp : bool
        if True, return the data directly
    type_v : string
        if 'real', only consider the real part of input ev 
        if 'imag', only consider the imag part of input ev 
        if 'abs', only consider the modula part of input ev 
     
    Returns
    -------
    velocity_choose1 : ndarray
        the velocity of those eigenvalues you choose(first range, small imaginary part)
      
    """
    type_list = [np.real(ev_connected_all), np.imag(ev_connected_all), np.abs(ev_connected_all)]
    velocitys = []
    for ev_connected in type_list:
        n_pars=ev_connected.shape[0]
        n_Evs=ev_connected.shape[1]
        lamda = np.linspace(0,np.pi,1500)[:n_pars]
        
        velocity = np.zeros((n_pars, n_Evs))
        for i in range(1, n_pars-1):
            for j in range(n_Evs):
                velocity[i,j] = (ev_connected[i+1, j]-ev_connected[i-1, j])/(lamda[i+1]-lamda[i-1])
        
        velocity_choose1 = np.array([])
        
        if rescale:
            mit = np.mean(velocity[1:-1,:], axis=0)
            mit_m = np.zeros_like(velocity)
            for i in range(1, n_pars-1):
                mit_m[i,:] = mit
            
            velocity_c = velocity-mit_m      #vc = v-<v>_lambda (<> means average in all paramters)
            var = np.zeros_like(velocity)
            mm0 = 5
            for i in range(1, n_pars-1):
                for j in range(mm0, n_Evs-mm0):
                    var[i, j] = np.mean(velocity_c[i, j-mm0:j+mm0]**2)    #var = <vc**2>_E (<> means local average in nearby energy)
            
            velocity_final = velocity_c[1:n_pars-1, mm0:n_Evs-mm0]/np.sqrt(var[1:n_pars-1, mm0:n_Evs-mm0])  #vf = vc/np.sqrt(var)
            ev_connected_cut = ev_connected_all[1:n_pars-1, mm0:n_Evs-mm0]
        else:
            velocity_final = velocity[1:n_pars-1, :]
            ev_connected_cut = ev_connected_all[1:n_pars-1, :]
        
        if flag_exp:
            velocitys.append(velocity_final)
        else:
            pick_range1 = (ev_connected_cut.real>Erange[0]) & (ev_connected_cut.real<Erange[1]) \
                        & (ev_connected_cut.imag>cutRange[0]) & (ev_connected_cut.imag<cutRange[1])
            velocity_choose1 = np.append(velocity_choose1, np.array(velocity_final[pick_range1]))
            velocitys.append(velocity_choose1)
    return velocitys

def Heff_ei(fixHrand1, fixHrand2, M, N, kappa, lamda): 
    """
    This function can calculate the eigenvalues varying the lambda
    """
    H0 = r.createRmtHWithSeed(fixHrand1, N, beta=1)
    H1 = r.createRmtHWithSeed(fixHrand2, N, beta=1)
    V = r.osys.channels.create_channels_with_given_coupling(
                M, N, kappa,
                random_container=False)
    ev = np.zeros((len(lamda), N), dtype=complex)
    for i in range(len(lamda)):
        heff = r.set_up_H_eff(H0 * np.cos(lamda[i]) + H1 * np.sin(lamda[i]), V)
        ei_heff = np.linalg.eigvals(heff)
        ev[i,:] = ei_heff
        
    return ev
    
def level_connect(ev, fixHrand1, fixHrand2, M, N, kappa, lamda):
    """
    This function tries to connect the eigenvalues belong to different dynamics

    Parameters
    ----------
    ev : ndarray
        eigenvalues matrix, ev.shape[0] is len(lambda), ev.shape[0] is N
    fixHrand1 : integer
        the value choose for H0
    fixHrand2 : integer
        the value choose for H1
    M : integer
        number of channels to set up
    N : integer
        size of the Hamiltonian
    kappa : scalar
        Describes the coupling in H_eff and determines the variance of
        the used Gaussian distribution.
    lamda : ndarray
        the parameter used to do the perturbation
        H0 * np.cos(lamda[i]) + H1 * np.sin(lamda[i])
    
    Returns
    -------
    ev_connected : ndarray
        sorted eigenvalues matrix
    
    """
    # orginal H and V information
    H0 = r.createRmtHWithSeed(fixHrand1, N, beta=1)
    H1 = r.createRmtHWithSeed(fixHrand2, N, beta=1)
    V = r.osys.channels.create_channels_with_given_coupling(
                M, N, kappa,
                random_container=False)
    
    evs=np.sort(ev, axis=1)
    n_Evs=evs.shape[1]
    n_pars=evs.shape[0]
    ev_connected=np.zeros_like(evs)
    ev_connected[0,:]=evs[0,:]
    for i in range(1,n_pars):
        ind_array=np.zeros(n_Evs, dtype=int)-n_Evs-10
        for iE in range(n_Evs):
            ind=np.argmin(np.abs(ev_connected[i-1,iE]-evs[i,:]))
            ind_array[iE]=ind
        values, counts = np.unique(ind_array, return_counts=True)
        ev_connected[i,:]=evs[i,ind_array]
        if counts.max() != 1:
            clear=True
            two_repeat_arg = np.where(ev_connected[i,:] == list_duplicates(ev_connected[i,:])[0])[0]
            while clear:
                print("Not unique smallest distances between lines {} and {}!!!, max={}".format(i-1,i,counts.max()))
                nmid=5
                while counts.max() != 1:
                    print(f'{nmid}')
                    lamda_mid = np.linspace(lamda[i-1], lamda[i], nmid)
                    ei_heff_midstep = np.zeros((nmid, n_Evs), dtype=complex)
                    for j in range(nmid):
                        heff_midstep = r.set_up_H_eff(H0 * np.cos(lamda_mid[j]) + H1 * np.sin(lamda_mid[j]), V)
                        ei_heff_midstep[j,:] = np.sort(np.linalg.eigvals(heff_midstep))
                    ev_connected_mid = np.zeros_like(ei_heff_midstep)
                    ev_connected_mid[0,:]=ev_connected[i-1,:]
                    for ji in range(1,nmid):
                        ind_array_mid=np.zeros(n_Evs, dtype=int)-n_Evs-10
                        for jiE in range(n_Evs):
                            ind=np.argmin(np.abs(ev_connected_mid[ji-1,jiE]-ei_heff_midstep[ji,:]))
                            ind_array_mid[jiE]=ind
                        values, counts = np.unique(ind_array_mid, return_counts=True)
                        ev_connected_mid[ji,:]=ei_heff_midstep[ji,ind_array_mid]
                    if counts.max() != 1:
                        nmid+=10
                    else:
                        corr_arg = ind_array_mid[two_repeat_arg]
                        ev_connected[i,two_repeat_arg] = ei_heff_midstep[-1, corr_arg]
                        # break
                        rep = list_duplicates(ev_connected[i,:])
                        if rep:
                            two_repeat_arg = np.where(ev_connected[i,:] == rep[0])[0]
                            values, counts = np.unique(ev_connected[i,:], return_counts=True)
                            clear=True
                        else:
                            clear=False
    return ev_connected

def list_duplicates(seq):
  seen = set()
  seen_add = seen.add
  # adds all elements it doesn't know yet to seen and all other to seen_twice
  seen_twice = set( x for x in seq if x in seen or seen_add(x) )
  # turn the set into a list (as requested)
  return list( seen_twice )

def generate_level_dynamics(M, N, kappa, n_lambda):
    fixHrand1 = random.randint(1,900000000)
    fixHrand2 = random.randint(1,900000000)
    index_lamba = random.randint(0,1500-n_lambda-50)
    lamda = np.linspace(0,np.pi,1500)[index_lamba:index_lamba+n_lambda]
    ei = Heff_ei(fixHrand1, fixHrand2, M, N, kappa, lamda)
    level_dy = level_connect(ei, fixHrand1, fixHrand2, M, N, kappa, lamda)
    level_argsort = np.argsort(np.abs(np.real(level_dy[0])))
    level_sort = level_dy[:, level_argsort]
    return level_sort
    
    
    