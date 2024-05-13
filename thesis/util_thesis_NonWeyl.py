#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:02:59 2021

@author: kuhl
"""
import platform
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from junjie.graph.num.resonances import extract_func_peak
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

computer_name=platform.node()

if computer_name=='plukh0041':
    path_base = '/home/kuhl/tmp/'
elif computer_name=='pwluj0041':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/mythesis/chapterNonWeyl/'
elif computer_name=='LAPTOP-K2TVHSG9':
    path_base = 'd:/onedrive/OneDrive - Université Nice Sophia Antipolis/Nice/mythesis/chapterNonWeyl/'
    
path_data = path_base + 'data/'
path_figs = path_base + 'figs/'

def useTex(_useTex, xs=15, ys=None):
    #xs=15, size=12 for phd thesis
    #
    if ys is None:
        ys=xs/((1+5**.5)/2)
    inch2cm=2.54
    if _useTex:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rc('font', size=12)
        plt.rc('figure', figsize= [xs/inch2cm, ys/inch2cm])
    else:
        plt.rc('text', usetex=False)
        plt.rc('font', family='sans-serif')
        plt.rc('font', size=10)
        plt.rc('figure', figsize= [xs, ys])
    

def spectrum_cal(k, example, flag_res = False):
    vn_eff, vna_eff = example()
    S = vna_eff.calcS(k)
    h = create_h(vn_eff,k)
    ei = get_ev_h(h,k)
    
    if flag_res:
        kr_cal = [80, 88, 4]
        ki_estimate = [-1, 0]
        poles = extract_func_peak(kr_cal, ki_estimate, graph_info=None,
                kr_tolrance=0.2, ki_tolrance=0.05, cal_func=vna_eff.calcS, extract_type='poles',
                n_discrete=500)
        return S, ei, poles
    else:
        return S, ei
    
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)
    
    return newcmap
    
    
    