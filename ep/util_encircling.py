# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 13:39:10 2023

@author: jlu
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

def plot_Riemann_surface_real(xx, yy, poles, r, encircling_type,
                              xlabel='', ylabel=''):
    poles = np.sort(poles, axis=2)

    fig = plt.figure() 
    ax = fig.add_subplot( projection='3d') 
    
    w = poles[:,:,0]
    z_data, c_data = w.real, -w.imag 
    c_data1 = (c_data - c_data.min()) / c_data.ptp() 
    colors1 = cm.viridis(c_data1) 
    
    w = poles[:,:,1]
    z_data2, c_data2 = w.real, -w.imag 
    c_data3 = (c_data2 - c_data2.min()) / c_data2.ptp() 
    colors2 = cm.viridis(c_data3) 
    
    clim0 = [np.min([c_data, c_data2]), np.max([c_data, c_data2])]
    
    surf = ax.plot_surface(xx, yy, z_data, facecolors=colors1,
                        clim=clim0)
    surf1 = ax.plot_surface(xx, yy, z_data2, facecolors=colors2,
                        clim=clim0)
    
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel)   
    ax.set_zlabel(r'$\mathrm{Re}(k)$')  
    cbar_ax = fig.add_axes([0.10, 0.26, 0.03, 0.5])
    cbar=fig.colorbar(mappable=surf, ax=ax, cax=cbar_ax)
    # cbar.ax.set_title(r'$-\mathrm{Im}(k)$', pad=6)
    
    if r:
        mask, mask_arg_to_sort = encircling_path(xx, yy, poles, r, encircling_type)
        poles_mask = poles[mask,:]
        xx = xx[mask]
        yy = yy[mask]
        ax.plot3D(xx, yy, poles_mask[:,0].real, ls=' ', marker='.', zorder=100)
        ax.plot3D(xx, yy, poles_mask[:,1].real, ls=' ', marker='.', zorder=100)
        return mask, mask_arg_to_sort, ax
    else:
        return None, None, ax
    
def plot_Riemann_surface_imag(xx, yy, poles, r, encircling_type,
                              xlabel='', ylabel=''):
    arg_poles = np.argsort(np.imag(poles), axis=2)
    poles =  np.take_along_axis(poles, arg_poles, axis=2)
    
    fig = plt.figure() 
    ax = fig.add_subplot( projection='3d') 
    
    w = poles[:,:,0]
    z_data, c_data = -w.imag, w.real 
    c_data1 = (c_data - c_data.min()) / c_data.ptp() 
    colors1 = cm.viridis(c_data1) 
    
    w = poles[:,:,1]
    z_data2, c_data2 = -w.imag, w.real 
    c_data3 = (c_data2 - c_data2.min()) / c_data2.ptp() 
    colors2 = cm.viridis(c_data3) 
    
    clim0 = [np.min([c_data, c_data2]), np.max([c_data, c_data2])]
    
    surf = ax.plot_surface(xx, yy, z_data, facecolors=colors1,
                        clim=clim0)
    surf1 = ax.plot_surface(xx, yy, z_data2, facecolors=colors2,
                        clim=clim0)
    
    ax.set_xlabel(xlabel)  
    ax.set_ylabel(ylabel)    
    ax.set_zlabel(r'$\mathrm{-Im}(k)$')  
    cbar_ax = fig.add_axes([0.10, 0.26, 0.03, 0.5])
    cbar=fig.colorbar(mappable=surf, ax=ax, cax=cbar_ax)
    # cbar.ax.set_title(r'$\mathrm{Re}(k)$', pad=6)
    
    if r:
        mask, mask_arg_to_sort = encircling_path(xx, yy, poles, r, encircling_type)
        poles_mask = poles[mask,:]
        xx = xx[mask]
        yy = yy[mask]
        ax.plot3D(xx, yy, -poles_mask[:,0].imag, ls=' ', marker='.', zorder=100)
        ax.plot3D(xx, yy, -poles_mask[:,1].imag, ls=' ', marker='.', zorder=100)
        return mask, mask_arg_to_sort, ax
    else:
        return None, None, ax

def encircling_path(xx, yy, poles, r, encircling_type):
    """
    3 ways to encircling around the EP
    
    encircling_type='c': 
        create a circle around EP point, 
        but the direction information is not avalible,
        and the input r is the radius
    
    encircling_type='s': 
        create a square around EP point, 
        the direction information is avalible,
        and the input r is the x/2 of a square
    
    encircling_type='r': 
        create a rectangular around EP point, 
        the direction information is avalible,
        and the input r=(a,b,c,d)
        (a,b) is the starting point, and (c,d) is the length and width of it

    """
    dis_poles = np.abs(poles[:,:,0]-poles[:,:,1])
    ind_ep = np.argwhere(dis_poles==np.min(dis_poles))[0]
    print(f'Ep: {ind_ep}')
    xx1, yy1 = np.meshgrid(np.arange(np.shape(xx)[1]), np.arange(np.shape(xx)[0]))
    
    if encircling_type == 'c':
        circle = (xx1 - ind_ep[1]) ** 2 + (yy1 - ind_ep[0]) ** 2
        mask = np.logical_and(circle < ((r+0.5)**2), circle > ((r-0.5)**2))
        mask_arg_to_sort = None
    elif encircling_type == 's':
        mask = np.zeros_like(xx, bool)
        mask_arg_to_sort = np.zeros_like(xx)
        mask[ind_ep[0]-r:ind_ep[0]+r, ind_ep[1]-r]=True
        mask[ind_ep[0]+r, ind_ep[1]-r:ind_ep[1]+r]=True
        mask[ind_ep[0]-r+1:ind_ep[0]+r+1, ind_ep[1]+r][::-1]=True
        mask[ind_ep[0]-r, ind_ep[1]-r+1:ind_ep[1]+r+1][::-1]=True
        
        mask_arg_to_sort[ind_ep[0]-r:ind_ep[0]+r, ind_ep[1]-r]=np.arange(2*r)
        mask_arg_to_sort[ind_ep[0]+r, ind_ep[1]-r:ind_ep[1]+r]=np.arange(2*r, 4*r)
        mask_arg_to_sort[ind_ep[0]-r+1:ind_ep[0]+r+1, ind_ep[1]+r][::-1]=np.arange(4*r, 6*r)
        mask_arg_to_sort[ind_ep[0]-r, ind_ep[1]-r+1:ind_ep[1]+r+1][::-1]=np.arange(6*r, 8*r)
        
        mask_arg_to_sort = np.argsort(mask_arg_to_sort[mask])
    elif encircling_type == 'r':
        (a,b,c,d)=r
        mask = np.zeros_like(xx, bool)
        mask_arg_to_sort = np.zeros_like(xx)
        mask[a:a+c,b]=True
        mask[a+c, b:b+d]=True
        mask[a+1:a+c+1, b+d][::-1]=True
        mask[a, b+1:b+d+1][::-1]=True
        
        mask_arg_to_sort[a:a+c,b]=np.arange(c)
        mask_arg_to_sort[a+c, b:b+d]=np.arange(c, c+d)
        mask_arg_to_sort[a+1:a+c+1, b+d][::-1]=np.arange(c+d, 2*c+d)
        mask_arg_to_sort[a, b+1:b+d+1][::-1]=np.arange(2*c+d, 2*c+2*d)
        
        mask_arg_to_sort = np.argsort(mask_arg_to_sort[mask])
        
    return mask, mask_arg_to_sort

    