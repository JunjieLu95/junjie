# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 10:40:30 2023

@author: jlu
"""

import numpy as np
from mesopylib.instrument_control.UsbBase import UsbBase
from mesopylib.instrument_control.SG_SiglentSDG6032X import SG_SiglentSDG6032X
from mesopylib.instrument_control.SG_Anritsu_68047C import SG_Anritsu_68047C
from mesopylib.instrument_control.SA_FSU26 import SA_FSU26
import matplotlib.pyplot as plt

sgSL = SG_SiglentSDG6032X()
sgA=SG_Anritsu_68047C(GPIBAddr=5, GPIBBoard=0)
sa=SA_FSU26(GPIBAddr=20, GPIBBoard=0)

#%%
frq_singal = 5000
amp = 10
amp_offset = 5
phase = 0
sym = 0

frq_center_set = 1
frq_center = 1.00000052
span_frq = 50
nPoints = 2501
vidBandWidth = 300
resBandWidth = 10
reference = 0

frq = np.linspace(frq_center-span_frq/1e6/2, frq_center+span_frq/1e6/2, nPoints)
savePath = "/home/qc/PowerFolders/Quantenchaos/High_sym_NMR/measurements/Junjie_Lu/230710_diode_sawtooth/"
fn =savePath+"5kHz_sawtooth_amp{}_off{}_sym{}.npy".format(amp, amp_offset, sym)

#%%
#sgSL.set_shape(1, 'RAMP')
sgSL.set_shape(1, 'SINE')
sgSL.set_frq(1, frq_singal)
sgSL.set_amplitude(1, amp)
sgSL.set_amplitude_offset(1, amp_offset)
sgSL.set_phase(1, phase)
#sgSL.set_symmetry(1, sym)
sgSL.turn_on_output(1)
#sgSL.turn_off_output(1)

sgA.setTimeOut(10000)
sgA.setFrq(frq_center_set)
sgA.setPower(-20)
print(( 'Minimal Frequency in Hz:', sgA.getMinimalFrq()))
print(( 'Maximal Frequency in Hz:', sgA.getMaximalFrq()))
print(( 'Actual set Frequency in GHz:', sgA.getFrq()))
print(( 'Actual power in dBm:', sgA.getPower()))

tmpStr="%10.8f" % frq_center
outstr= ':SENS:FREQUENCY:CENTER '+tmpStr+'GHz;'
sa.GPIBwrite(outstr)
        
tmpStr="%6.3f" % span_frq
outstr=':SENS:FREQUENCY:SPAN '+ tmpStr +'KHz;'
sa.GPIBwrite(outstr)

outstr='SWE:POIN '+str(nPoints);
sa.GPIBwrite(outstr)

tmpStr="%10.3f" % resBandWidth
outstr='BAND:RES '+tmpStr+'Hz;' # RBW=1kHz
sa.GPIBwrite(outstr)

tmpStr="%10.3f" % vidBandWidth
outstr='BAND:VID '+tmpStr+'Hz;' # VBW=1kHz
sa.GPIBwrite(outstr)

tmpStr="%6.3f" % reference
outstr='DISP:WIND:TRAC:Y:RLEV '+ tmpStr+'dBm';
sa.GPIBwrite(outstr)

sa.GPIBwrite('INIT:CONT OFF;:INIT')
sa.GPIBwrite('*WAI')
data = sa._Sa_ReceiveData()

plt.plot(frq, data)

np.save(fn, np.array([frq, data]))


