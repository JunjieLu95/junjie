# -*- coding: utf-8 -*-
"""
Class to control Signal Generator Keysight/Agilent 33621A:
Keysight 33621A 120 MHz Function / Arbitrary Waveform Generator
Details see manual: http://rfmw.em.keysight.com/bihelpfiles/Trueform/webhelp/DE/Default.htm
or 33500-90911.pdf
6C0633120A_USERSGUIDE_ENGLISH.pdf


Created on Wed Aug 15 12:45:35 2018

@author: Tobias Hofmann
"""

import os
import time
import numpy as np
from mesopylib.instrument_control.UsbBase import UsbBase
from mesopylib.load_save.fileDialogs import saveDialog, overwriteDialog
from mesopylib.instrument_control.UsbBase import UsbBase

class SgError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SG_SiglentSDG6032X(UsbBase):
    """
    Class to control Signal Generator SDG6032X:
    Arbitrary Waveform Generator
    Details see manual:
    SDG_Programming-Guide_PG02-E05A-12.pdf 
    """

    def __init__(self, usb_addr="62700::4353::SDG6XFCX7R0338::0::INSTR", usb_board=0): #Windows USB-Adresse in Hex nicht dec
        # defining variables which are also properties (without _)
        super(SG_SiglentSDG6032X, self).__init__(usb_addr=usb_addr, usb_board=usb_board)
        self.dataType = 0
        self.instrStr = 'Siglent Technologies, 6032X'
        self.instrStrShort = 'SDG6032X'
        self.info = None
        self._frq = None  # in Hz
        self._amp = None
        self.data = None
        self._shape = None

    def __enter__(self,):
        return self
    
    def __exit__(self, *args):
        self.output = 0
        self.USBInterface.close()
    # -----------------------------------------
    # Properties

    def reset(self):
        """ Reset and Initialize device"""
        self.USBwrite('*RST;')  # Reset the device and clear status
        self.USBwrite('*CLS;')  # Reset the device and clear status

    def init(self):
        """initialize Signal Analyzer to default state"""

    def set_shape(self, ch, value='SINE'):
        """ Set the shape type of the signal, i.e. sinusoidla, rectangular,...:
        {SINE, SQUARE, RAMP, PULSE, NOISE, ARB, DC, PRBS, IQ}
        """
        outstr = 'C%s:BSWV WVTP,%s' % (ch, value)
        self.USBwrite(outstr)
        
    # Frequency Controll
    def set_frq(self, ch, value):
        """set frequency in Hz"""  
        outstr = 'C%s:BSWV FRQ,%s' % (ch, value)
        self.USBwrite(outstr)
    
    # Amplitude Control
    def set_amplitude(self, ch, value):
        """set amplitude in Volts"""
        self._amp = value
        outstr = 'C%s:BSWV AMP,%s' % (ch, value)
        self.USBwrite(outstr)
        
    # Offset Control
    def set_amplitude_offset(self, ch, value):
        """set offset of the amplitude in Volts"""
        outstr = 'C%s:BSWV OFST,%s' % (ch, value)
        self.USBwrite(outstr)
        
    # Phase Control
    def set_phase(self, ch, value):
        """set phase in degree"""
        outstr = 'C%s:BSWV PHSE,%s' % (ch, value)
        self.USBwrite(outstr)
        
    # Symmetry Control
    def set_symmetry(self, ch, value):
        """Symmetry of RAMP. The unit is "%". Only settable when WVTP is RAMP."""
        outstr = 'C%s:BSWV SYM,%s' % (ch, value)
        self.USBwrite(outstr)
    
    def turn_on_output(self, ch):
        """Turn on output."""
        outstr = 'C%s:OUTP ON' % (ch)
        self.USBwrite(outstr)
        
    def turn_off_output(self, ch):
        """Turn off output."""
        outstr = 'C%s:OUTP OFF' % (ch)
        self.USBwrite(outstr)
    
    def get_all_parameters(self, ch):
        """ Get the shape type of the signal, i.e. sinusoidla, rectangular,...:
            {SINE, SQUARE, RAMP, PULSE, NOISE, ARB, DC, PRBS, IQ}
        """
        outstr = 'C%s:BSWV?'% (ch)
        result = self.USBquery(outstr)
        self._shape = result
        return result

    # shape = property(get_shape, set_shape)


    # def get_amplitude(self):
    #     """get amplitude in Volts"""
    #     outstr = 'SOURCE:VOLTAGE?'
    #     result = self.USBquery(outstr)
    #     self._amp = result
    #     return float(result)

    # amp = property(get_amplitude, set_amplitude)

    

    # def get_amplitude_offset(self):
    #     """get offset of the amplitude"""
    #     outstr = 'SOURCE:VOLTAGE:OFFSET?'
    #     result = self.USBquery(outstr)
    #     return float(result)

    # offset = property(get_amplitude_offset, set_amplitude_offset)

    # # Voltage Units
    # def set_amplitude_unit(self, value):
    #     """set offset of the amplitude in Volts:
    #     {VPP|VRMS|DBM|DEFault}
    #     """
    #     outstr = 'SOURCE:VOLTAGE:UNIT {0}'.format(value)
    #     self.USBwrite(outstr)

    # def get_amplitude_unit(self):
    #     """get offset of the amplitude: 
    #     {VPP|VRMS|DBM|DEFault}
    #     """
    #     outstr = 'SOURCE:VOLTAGE:UNIT?'
    #     result = self.USBquery(outstr)
    #     return float(result)

    # amp_unit = property(get_amplitude_unit, set_amplitude_unit)

    #   PRBS Controll
    # def set_bitrate(self, val):
    #     """set bitrate of the PRBS in bits/s
    #     [SOURce[1|2]:]FUNCtion:PRBS:BRATe{<bit_rate>|MINimum|MAXimum|DEFault}"""
    #     outstr = "SOUR:FUNC:PRBS:BRAT {0}".format(val)
    #     self.USBwrite(outstr)

    # def get_bitrate(self):
    #     outstr = "SOUR:FUNC:PRBS:BRAT?"
    #     result = self.USBquery(outstr)
    #     return float(result)

    # bitrate = property(get_bitrate, set_bitrate)

    # def set_PRBS_data(self, val):
    #     """[SOURce[1|2]:]FUNCtion:PRBS:DATA <sequence_type>
    #     [SOURce[1|2]:]FUNCtion:PRBS:DATA?
    #     val=PN# (PN3 bis PN9) ode PN## (PN10 bis PN32), wobei jedes # einer Ziffer entspricht.
    #     Legt die (PRBS) - Art der pseudozufälligen Binärfolge fest.
    #     Die Einstellung des Sequenztyps legt Länge und Feedbackwerte wie unten beschrieben fest"""
    #     outstr = "SOUR:FUNC:PRBS:DATA {0}".format(val)
    #     self.USBwrite(outstr)

    # def get_PRBS_data(self):
    #     outstr = "SOUR:FUNC:PRBS:DATA?"
    #     result = self.USBquery(outstr)
    #     return result

    # prbs_data = property(get_PRBS_data, set_PRBS_data)

    # # Output_Load
    # def set_load(val):
    #     return ()

    # ##### XMLDAT
    # def xmlSaveData(self, f):
    #     import mesopylib.load_save.xmlOwn as xmlOwn
    #     f.write('<SignalGneratorData>\n')
    #     data = self.data
    #     xmlOwn.xmlSaveBinaryArray(f, data, name='Waveform')
    #     f.write('</SignalGneratorData>\n')

    # def xmlSaveHeader(self, f):
    #     from mesopylib.load_save.xmlOwn import xmlCreateKeyValue
    #     import xml.dom.minidom as dom
    #     base_tag = dom.Element('SignalGeneratorHeader')
    #     base_tag.setAttribute('Version', '0.1')
    #     base_tag.setAttribute('Name', 'KS33621A')
    #     f_tag = dom.Element('Frequency')
    #     xmlCreateKeyValue('Frequency', '{:E}'.format(np.float(self.frq)), attr=[('unit', 'Hz')], base_tag=f_tag)
    #     xmlCreateKeyValue('Amplitude', '{:E}'.format(np.float(self.amp)), attr=[('unit', 'dBm')], base_tag=f_tag)
    #     base_tag.appendChild(f_tag)
    #     outStr = self.info
    #     prop_tag = dom.Element("Properties")
    #     xmlCreateKeyValue('Info', outStr, base_tag=prop_tag)
    #     base_tag.appendChild(prop_tag)
    #     base_tag.writexml(f, "", "\t", "\n")

    # def xmlSave(self, filename=None, f=None, fileOverwrite=False, standard=None, cal_kit=None):
    #     open_flag = (f == None)
    #     if open_flag:
    #         if filename is None:
    #             filename = saveDialog(filename=filename, name=None, path=None, wildcard='*.xmldat',
    #                                   title='Save xml data file')
    #         while os.path.isfile(filename) and not fileOverwrite:
    #             reply = overwriteDialog(fileName=filename)
    #             if reply == 0:
    #                 filename = saveDialog(filename=filename, name=None, path=None, wildcard='*.xmldat',
    #                                       title='Save xml data file')
    #                 if filename == '':
    #                     reply = 1
    #                     return False
    #                 else:
    #                     # filename=filename[0]
    #                     fileOverwrite = True
    #             if reply == 2:  # Cancel pressed
    #                 print('No file saved!!!')
    #                 return False
    #             if reply == 1:  # Cancel pressed
    #                 fileOverwrite = True
    #         # create first line!
    #         self.fileName = filename
    #         f = open(filename, 'wb')
    #         f.write('<?xml version="1.0" ?>\n')
    #         f.write('<MesoNiceXMLFile Version="0.1">\n')
    #     txt = '<SignalGenerator>\n'
    #     f.write(txt)
    #     self.xmlSaveHeader(f)
    #     self.xmlSaveData(f)
    #     f.write('</SignalGenerator>\n')
    #     if open_flag:
    #         f.write('</MesoNiceXMLFile>')
    #         f.close()
    #     return True





if __name__ == "__main__":
    sgSL = SG_SiglentSDG6032X()

