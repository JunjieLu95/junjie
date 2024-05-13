#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 13:15:08 2021

@author: qc
"""

from VNA_single_meas import vna_single_meas
import numpy as np
import mesopylib.instrument_control.MotorPDS01 as psd
from mesopylib.instrument_control.VNAAgilent8720ES import VNAAgilent8720ES
import time
from mesopylib.instrument_control.SG_Keysight33621A import SG_Keysight33621A as sg

ps = psd.MotorPDS01()

### VNA Init
vna=VNAAgilent8720ES()
vna.reset()
vna.startFrq=0.1E9
vna.stopFrq=18.0E9
#        vna.startFrq=1.3E9
#        vna.stopFrq=2.1E9
#vna.power = -20
vna.numberOfTotalPoints=64001
vna.numberofPoints=64001
vna.nPointsPerWindow=1601
vna.nWindows=40
vna.iFBandWidth=1000
vna.averageFlag=False
vna.averageNum=0
vna.setTimeOut(timeout=200000L)
vna.calFileName = "cal20220110_1"
vna.sParameterList=['S11']
#        vna.sParameterList=["S11", 'S22']
#vna.sParameterList=['S11', "S12", "S21", "S22"]
#        vna.sParameterList=['S12', "S21"]
vna.set2Vna()
#
#Motor 4 is used in setup
ps.openPort()
ps.findOrigin(4)


time.sleep(20)

#ini_ps2 =58000

ps.findOrigin(2)
time.sleep(10)

#savePath = "./measurements/100122/NMR_Messung1.4-1.7GHz_ZirkShort_Ankoppl_Dynamisch_PS={}/"
#savePath = "./measurements/100122/hihg_sym_graph/static_AllesGleichLang_2cmHub_Ps2-0_Ps4_{}/"
savePath = "/home/qc/PowerFolders/Quantenchaos/High_sym_NMR/measurements/Junjie_Lu/Cablelength/"
filename ="Ygraph_workshop_cut_2ff.xmldat"
vna_single_meas(savePath, filename, fast=1, instr=vna)

#savePath = "./measurements/Junjie_Lu/Cablelength/"

#fileNamePattern = "Laenge_GrauMM+KabelDIAB+PS_100000+2ff.xmldat"

#fileNamePattern = "NMR_Messung_1.4-1.7GHz_4Window_ZirkShort_AnkopplungVerschlechtert_PS={}_Modfreq={}MHz.xmldat"

#        time.sleep(0.5)

        
        
#with sg as sg:
#    for rate in fliprate_array_scan:
#        sg.output = 1
#        sg.frq = rate
#        filename = fileNamePattern.format(rate)
#        time.sleep(1)
#        vna_single_meas(savePath, filename)                
#    ps.moveToAbs(4, 0)
#    time.sleep(10)
#################################################
#pos1_array_scan = np.linspace(0, 200000, 2, dtype = int)                
#pos_array_scan = np.linspace(0, 200000, 11, dtype = int)
#fliprate_array_scan = np.linspace(100000.0, 21100000.0, 26, dtype = float)
#with sg as sg:
##    for pos1 in pos1_array_scan:
##        sg.output = 1
##        ps.moveToAbs(2, pos1)
##        time.sleep(10)
#    for pos in pos_array_scan:
#        sg.output = 1
#        ps.moveToAbs(4, pos)
#        time.sleep(10)
#    for rate in fliprate_array_scan:
#        sg.output = 1
#        sg.frq = rate
#        filename = fileNamePattern.format(rate)
#        time.sleep(1)
#        vna_single_meas(savePath, filename)
# ##############################################   

    
    
#motor_pos_delay = 96700
#amp = 10
#off = 0
#shape = "PRBS"
#with sg() as sg:
##    offset_l = [0,1]
#    sg.shape = shape
#    sg.amp = amp
#    sg.offset = off
#    sg.output = 1
#    for rate in fliprate_array_scan:
#        sg.bitrate = rate
#        time.sleep(0.5)
#        filename = fileNamePattern.format(rate)
#        vna_single_meas(savePath, filename)
        
        

#    for pos in motor_pos_array_scan:
#        ps.moveToAbs(2, pos)
#        ps.moveToAbs(4, 200000-pos)
#        time.sleep(5)
#        for offset in offset_l:
#            fileName = fileNamePattern.format(offset, pos)
#            sg.offset = offset
#            sg.output = 1
#            vna_single_meas(savePath, fileName)