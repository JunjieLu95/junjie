# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:41:54 2023

@author: jlu
"""
import numpy as np
from mesopylib.extract.frq2 import frq2k
from junjie.ep.util_ep import cal_real_phaser

def data_dicts_no_ratio(i, j):
    if i==0 and j==0:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230526_Ygraph_pos_l2_47000_65000_scan1',
                'cal_name' : '230525cal_0.1-18_40Win',
                'fn_cal_data' : '230526_Ygraph_pos_l2_47000_65000_scan1.npy',
                'pos_l2' : np.linspace(47000, 65000, 7, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 401, dtype=np.int),
                'frq' : np.linspace(0.1, 18, 64001),
                'phis' : np.linspace(20, 120, 501),
                'fn_sp' : '230526_Ygraph_scan1_phi_20_120_sp.npy',
                'fn_poles' : '230526_Ygraph_scan1_phi_20_120_poles_HI.npy'
                }
    elif i==1 and j==0:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230531_Ygraph_pos_l2_59000_62000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230531_Ygraph_pos_l2_59000_62000_scan2.npy',
                'pos_l2' : np.linspace(59000, 62000, 151, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 101, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201), 
                'phis' : np.linspace(33*np.pi, 35*np.pi, 101),
                'fn_sp' : '230531_Ygraph_scan2_phi_33pi_35pi_sp.npy',
                'fn_poles' : '230531_Ygraph_scan2_phi_33pi_35pi_poles_HI.npy'
                }
    elif i==1 and j==1:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230531_Ygraph_pos_l2_59000_62000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230531_Ygraph_pos_l2_59000_62000_scan2.npy',
                'pos_l2' : np.linspace(59000, 62000, 151, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 101, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.11769019, 105.30618575, 101),
                'fn_sp' : '230531_Ygraph_scan2_phi_105.11_105.31_sp.npy',
                'fn_poles' : '230531_Ygraph_scan2_phi_105.11_105.31_poles_LZ1.npy',
                'fit_initial' : [-2.37951407e+00 +0.29956157j,  1.66934953e+02 +0.4378175j , 3.40712823e+00 -1.74158137j,  1.67068109e+02 +0.61964104j, 1.39304485e+01-19.57495826j, -8.92023187e-02 +0.11252083j]
                }
    elif i==2 and j==0:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230602_Ygraph_pos_l2_60000_61000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230602_Ygraph_pos_l2_60000_61000_scan2.npy',
                'pos_l2' : np.linspace(60000, 61000, 201, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 201, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201),
                'fit_krange' : (166,168),
                'phis' : np.linspace(33*np.pi, 35*np.pi, 101),
                'fn_sp' : '230602_Ygraph_scan2_phi_33pi_35pi_sp.npy',
                'fn_poles' : '230602_Ygraph_scan2_phi_33pi_35pi_poles_HI.npy'
                 }
    elif i==2 and j==1:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230602_Ygraph_pos_l2_60000_61000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230602_Ygraph_pos_l2_60000_61000_scan2.npy',
                'pos_l2' : np.linspace(60000, 61000, 201, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 201, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.11769019, 105.30618575, 101),
                'fn_sp' : '230602_Ygraph_scan2_phi_105.11_105.31_sp.npy',
                'fn_poles' : '230602_Ygraph_scan2_phi_105.11_105.31_poles_LZ.npy',
                'fit_initial' : [-2.11323420e+00 -0.47816313j,  1.66910677e+02 +0.44085647j,
                3.09599782e+00 -0.85205841j,  1.67098707e+02 +0.59284455j,
                1.24281506e+01-16.99347486j, -7.98136328e-02 +0.09718997j]
                }
    elif i==2 and j==2:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230602_Ygraph_pos_l2_60000_61000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230602_Ygraph_pos_l2_60000_61000_scan2.npy',
                'pos_l2' : np.linspace(60000, 61000, 201, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 201, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.174238858, 105.183663636, 51),
                'fn_sp' : '230602_Ygraph_scan2_phi_105.174_105.184_sp.npy',
                'fn_poles' : '230602_Ygraph_scan2_phi_105.174_105.184_poles_LZ.npy',
                'fit_initial' : [-2.14472021e+00 +1.29567305j,  1.66963906e+02 +0.43278164j,
                3.15055626e+00 -2.69609972j,  1.67025562e+02 +0.62923203j,
                1.23678449e+01-17.36505882j, -7.95612251e-02 +0.09948043j]
                }
    elif i==2 and j==3:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0.000855,
                'data_path' : '230602_Ygraph_pos_l2_60000_61000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230602_Ygraph_pos_l2_60000_61000_scan2.npy',
                'pos_l2' : np.linspace(60000, 61000, 201, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 201, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105, 105.6, 51),
                'fn_sp' : '230602_Ygraph_scan2_phi_105_105.6_sp_adjust.npy',
                'fn_poles' : '230602_Ygraph_scan2_phi_105_105.6_poles_LZ_adjust.npy',
                'fit_initial' :[ 2.00558364e+00 -0.56877561j,  1.67189029e+02 +0.58845274j,
                -1.08331944e+00 -0.71273881j,  1.66862499e+02 +0.41137563j,
                1.14811482e+01-19.00146619j, -7.41682869e-02 +0.10921638j]
                }
    elif i==2 and j==4:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0.000855,
                'data_path' : '230602_Ygraph_pos_l2_60000_61000_scan2',
                'cal_name' : '230601cal_7.73-8.21_2Win',
                'fn_cal_data' : '230602_Ygraph_pos_l2_60000_61000_scan2.npy',
                'pos_l2' : np.linspace(60000, 61000, 201, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 201, dtype=np.int),
                'frq' : np.linspace(7.73, 8.21, 3201),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.36, 105.6, 51),
                'fn_sp' : '230602_Ygraph_scan2_phi_105.36_105.6_sp_adjust.npy',
                'fn_poles' : '230602_Ygraph_scan2_phi_105.36_105.6_poles_LZ_adjust.npy',
                'fit_initial' :[ 2.52242802e+00 -2.93205105j,  1.67005272e+02 +0.65414234j,
               -1.50792220e+00 +1.49878503j,  1.66977993e+02 +0.42009493j,
                1.23687537e+01-17.68828991j, -7.96278511e-02 +0.10144067j],
                }
    elif i==3 and j==0:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230607_Ygraph_pos_l2_60100_60600_scan3',
                'cal_name' : '230602cal_7.92-8.016_1Win',
                'fn_cal_data' : '230607_Ygraph_pos_l2_60100_60600_scan3.npy',
                'pos_l2' : np.linspace(60100, 60600, 101, dtype=np.int),
                'pos_lphi' : np.linspace(113000, 133000, 201, dtype=np.int),
                'frq' : np.linspace(7.92, 8.016, 1601),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.11769019, 105.30618575, 201),
                'fn_sp' : '230607_Ygraph_scan3_phi_105.11_105.31_sp.npy',
                'fn_poles' : '230607_Ygraph_scan3_phi_105.11_105.31_poles_LZ.npy',
                'fit_initial' : [-2.91106319e+00 -1.62699415j,  1.66910578e+02 +0.48231782j,
                3.93021978e+00 +0.28279799j,  1.67077935e+02 +0.54859036j,
                1.26016385e+01-17.93021425j, -8.09724570e-02 +0.10274581j]
                }
    elif i==3 and j==1:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230607_Ygraph_pos_l2_60100_60600_scan3',
                'cal_name' : '230602cal_7.92-8.016_1Win',
                'fn_cal_data' : '230607_Ygraph_pos_l2_60100_60600_scan3.npy',
                'pos_l2' : np.linspace(60100, 60600, 101, dtype=np.int),
                'pos_lphi' : np.linspace(113000, 133000, 201, dtype=np.int),
                'frq' : np.linspace(7.92, 8.016, 1601),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.136539746, 105.174238858, 101),
                'fn_sp' : '230607_Ygraph_scan3_phi_105.136_105.174_sp.npy',
                'fn_poles' : '230607_Ygraph_scan3_phi_105.136_105.174_poles_LZ.npy',
                'fit_initial' : [-4.85353094e+00 -1.83317113j,  1.66936662e+02 +0.4943727j ,
                5.88173775e+00 +0.46718043j,  1.67044567e+02 +0.5456434j ,
                1.26462934e+01-17.9770555j , -8.12703877e-02 +0.10304489j]
                }
    elif i==4 and j==0:
        dicts = {'fn_length' : '230526_Ygraph_bonds.npy', 'len_adjust' : 0,
                'data_path' : '230620_Ygraph_ratio2_scan1',
                'cal_name' : '230620cal_0.1-18_40Win',
                'fn_cal_data' : '230620_Ygraph_ratio2_scan1.npy',
                'pos_l2' : np.linspace(0, 200000, 51, dtype=np.int),
                'pos_lphi' : np.linspace(0, 200000, 51, dtype=np.int),
                'frq' : np.linspace(0.1, 18, 64001),
                'fit_krange' : (166,168),
                'phis' : np.linspace(105.136539746, 105.174238858, 101),
                'fn_sp' : '230607_Ygraph_scan3_phi_105.136_105.174_sp.npy',
                'fn_poles' : '230607_Ygraph_scan3_phi_105.136_105.174_poles_LZ.npy',
                'fit_initial' : [-4.85353094e+00 -1.83317113j,  1.66936662e+02 +0.4943727j ,
                5.88173775e+00 +0.46718043j,  1.67044567e+02 +0.5456434j ,
                1.26462934e+01-17.9770555j , -8.12703877e-02 +0.10304489j]
                }
    # k = frq2k(dicts['frq'])*1e9
    # phaser = cal_real_phaser(dicts['pos_lphi'], dicts['len_adjust'])
    # pos_l2 = dicts['pos_l2']
    # pos_lphi = dicts['pos_lphi']
        
    return dicts