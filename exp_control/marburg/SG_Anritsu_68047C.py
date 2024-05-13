# -*- coding: utf-8 -*-
"""
Class to control the Anritsu 68047C Signal Generator via GPIB.

Created on Wed December 02 16:00:00 2014

@author: Kuhl
"""

from mesopylib.instrument_control.GPIB_Base import GPIB_Base

class SG_Anritsu_68047C(GPIB_Base):#, GPIBAddr=5, GPIBBoard=0):
    """
    Class to control the Anritsu 68047C Signal Generator via GPIB.
    Commands used are in GPIB mode (not in SCPI)!
    See Anritsu_Series 680XXC Synthesized CW Generator GPIB_Programmi.pdf
    """    
    def reset(self):
        """ Reset and Initialize device
        """
        #{ RST : reset 360B to default parameters
        #  FHI : sets data points to maximum (501) }
        self.GPIBwrite('RST FHI ')
    
    def GetInfo(self):
        """Get Info Str:\n
        Causes the CW generator to identify itself by sending the
        following parameter information over the bus; model number,
        low-end frequency, high-end frequency, minimum output power
        level, maximum output power level, software revision number,
        serial number, model prefix (A or B), and series (1 or 2). This
        command can be used to send parameter information to the
        controller automatically, thus relieving the operator from having
        to input the information manually. The string is 36 characters
        long.
        """
        result=self.GPIBquery('OI')
        return result        

    def setFrq(self, frq):
        """ frq in GHz """
        tmpStr="%9.6f" % frq #str(CWFreq:9:6);
        outstr='CF0 '+tmpStr+' GH' # Set start frequency
        self.GPIBwrite(outstr)

    def getMinimalFrq(self):
        """ return lowest setable frequency in Hz"""
        outstr='OFL'
        result=self.GPIBquery(outstr)
        # frequecny in MHz
        return float(result)*1e3
        
    def getMaximalFrq(self):
        """ return highest frequency in Hz"""
        outstr='OFH'
        result=self.GPIBquery(outstr)
        return float(result)*1e3

    def getFrq(self):
        """ return frequency in GHz """
        outstr='OF0'
        result=self.GPIBquery(outstr)
        return float(result)/1e3

#   Power commands:
    def setPower(self, power):
        """ set Power in dBm and go to continous """
        #LIN Selects linear power level operation 2-21
        #LOG Selects logarithmic power level operation 2-21
        #RF0 Turns off the RF output 2-22
        #RF1 Turns on the RF output 2-22
        tmpStr="%4.1f" % power
        outstr= 'LOG '+'RF0 L1 '+tmpStr+'DM RF1';
        self.GPIBwrite(outstr)

    def getPower(self):
        """ get Power """
        outstr='OL0'
        result=self.GPIBquery(outstr)
        return float(result)

    def getPowerOffset(self):
        """ OLO Returns the Level Offset power value (in dB when in log mode;
            in mV when in linear mode) to the controller """
        outstr='OLO'
        result=self.GPIBquery(outstr)
        return result

    def getState(self):    
        # at the moment not working as it returns non ACSCII characters
        """ SAF Outputs the current instrument setup to the controller 
        10 ! GET OI
        20 ! Gets the output id string
        30 ! from a 68XXXX Synthesizer
        100 OUTPUT 705; OI
        110 DIM A$[36]
        120 ENTER 705; A$
        130 M$=A$[1,2] ! Model
        140 M1$=A$[3,4] ! Model Number
        150 F1$=A$[5,9] ! Freq Low
        160 F2$=A$[10,14] ! Freq High
        170 L2$=A$[15,20] ! Min Power
        180 L1$=A$[21,24] ! Max Power
        190 S$=A$[25,28] ! Software Ver
        200 S1$=A$[29,34] ! Serial Number
        210 P$=A$[35,35] ! Model Prefix
        220 S2$=A$[36] ! Series
        """
        outstr='SAF'
        result=self.GPIBqueryBin(outstr)
        return result

    def getGPIB_TerminationStatus(self):
        """ OWT Returns the GPIB termination status to the controller. (0=CR; 1=CRLF)"""
        outstr='OWT'
        result=self.GPIBquery(outstr)
        return result


if __name__ == "__main__":
    #gpib=GPIB_Base()
    sgA=SG_Anritsu_68047C(GPIBAddr=5, GPIBBoard=0)
    sgA.setTimeOut(1000)
    sgA.setFrq(10)
    sgA.setPower(1)
    print(( 'Minimal Frequency in Hz:', sgA.getMinimalFrq()))
    print(( 'Maximal Frequency in Hz:', sgA.getMaximalFrq()))
    print(( 'Actual set Frequency in GHz:', sgA.getFrq()))
    print(( 'Actual power in dBm:', sgA.getPower()))
    pass