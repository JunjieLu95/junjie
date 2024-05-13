# -*- coding: utf-8 -*-
"""
Created on Wed December 02 16:00:00 2014

@author: Kuhl
"""
import os
import numpy as np
from mesopylib.load_save.fileDialogs import loadDialog, saveDialog, overwriteDialog
from mesopylib.instrument_control.GPIB_Base import GPIB_Base

class SaError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
        
class SA_FSU26(GPIB_Base):#, GPIBAddr=5, GPIBBoard=0):

    def __init__(self, GPIBAddr=20, GPIBBoard=0):
        # defining variables which are also properties (without _)
        super(SA_FSU26, self).__init__(GPIBAddr=GPIBAddr, GPIBBoard=GPIBBoard)
        self.dataType=0;
        self.filter='unknown';
        self.RBWType='NORM';
        self.VBWType='NORM';
        self.SAStr='unknown';
        self.SAShortStr='unknown';
        self.reference=0;
        self.attenuator=10;
        self.attenuation=10 # in dB
        self.detector='POSITIVE';
        self.SAStr = 'FSU 26 (Rohde & Schwarz)';
        self.FSAShortStr = 'FSU26';
        self.resBandWidth=100e3 # in Hz
        self.vidBandWidth=300e3 # in Hz
        self._nPoints=313; #  //nPoints: 155,313,625,1251,2501,5001,10001
        self.startTime=None
        self.endTime=None
        self.dataType = 2;
        self._startFrq=10e9 # in Hz
        self._stopFrq=11e9 # in Hz
        self.data=None
        
#-------------------------------properties        
    def _set_startFrq(self, value):
        self._startFrq=value
    def _get_startFrq(self):
        return self._startFrq
    startFrq = property(fget=lambda self: self._get_startFrq(), 
                        fset=lambda self, value: self._set_startFrq(value))

    def _set_stopFrq(self, value):
        self._stopFrq=value
    def _get_stopFrq(self):
        return self._stopFrq
    stopFrq = property(fget=lambda self: self._get_stopFrq(), 
                        fset=lambda self, value: self._set_stopFrq(value))
    
    def _set_nPoints(self, value):
        values=np.array([155,313,625,1251,2501,5001,10001])
        if value in values:
            self._nPoints=value
        else :
            raise SaError('nPoint Value not correct')
            
    def _get_nPoints(self):
        return self._nPoints
    nPoints = property(fget=lambda self: self._get_nPoints(), 
                        fset=lambda self, value: self._set_nPoints(value))

    def _set_centerFrq(self, value):
        span=self.span
        self.startFrq=value-span
        self.stopFrq=value+span
    def _get_centerFrq(self):
        return (self.stopFrq+self._startFrq)/2.        
    centerFrq = property(fget=lambda self: self._get_centerFrq(), 
                        fset=lambda self, value: self._set_centerFrq(value))

    def _set_span(self, value):
        centerFrq=self.centerFrq
        self.startFrq=centerFrq-value
        self.stopFrq=centerFrq+value
    def _get_span(self):
        return self.stopFrq-self._startFrq
    span = property(fget=lambda self: self._get_span(), 
                        fset=lambda self, value: self._set_span(value))

    def setCenterSpan(self, inCenterFreq=None, inSpan=None):
        """calculates from center and span the start and stop frequencies\n
           centerfrq in Hz Span in Hz
        """
        self.startFrq = inCenterFreq-inSpan/2.;
        self.stopFrq = inCenterFreq+inSpan/2.;

    def reset(self):
        """ Reset and Initialize device"""
        self.GPIBwrite(':INST SA;') # Set the instrument mode to spectrum analysis
        self.GPIBwrite('*RST;')     # Reset the device and clear status
        self.GPIBwrite('*CLS;')     # Reset the device and clear status
    
    def init(self):
        """initialize Signal Analyzer to default state"""
        self.GPIBwrite('SYST:DISP:UPD ON') # Bildschirmdarstellung ein
        self.GPIBwrite('DISP:FORM SING') # Full Screen Darstellung
        self.GPIBwrite('DISP:WIND1:SEL') # Active Screen A
        self.GPIBwrite('INIT:CONT OFF') # Full Screen Darstellung
        self.GPIBwrite('INP:ATT 0dB') # Input Attenuator auf 0dB
        self.GPIBwrite('BAND:RES 100KHz') # Resolution bandwidth
        #//  IeeeBus.Send(IEC_Adr, 'BAND:VID 1MHz') # Video bandwidth
        self.GPIBwrite('BAND:VID:AUTO ON') # Video bandwidth
        self.GPIBwrite('SWE:TIME:AUTO ON') # Sweeptime
        #//  IeeeBus.Send(IEC_Adr, 'SWE:TIME 100ms') # Sweeptime
        self.GPIBwrite('SWE:COUN 1') # Number of sweeps
        self.GPIBwrite('SENS:FREQUENCY:CENTER 10GHz;') # Center Frequenz einstellen
        self.GPIBwrite('SENS:FREQUENCY:SPAN 100MHz;') # Span Frequenz einstellen
#{  Detector:
#FSU [SENSe<1|2>:]DETector<1...3>[:FUNCtion]
#APEak |NEGative | POSitive | SAMPle | RMS | AVERage | QPEak
#number of traces restricted to 3; detector settings correspond to selected screen
#}
#{
#DETECTOR -- AUTO SELECT [SENSe:]DETector[:FUNCtion]:AUTO ON | OFF
#DETECTOR AUTOPEAK [SENSe:]DETector[:FUNCtion] APEak
#DETECTOR MAX PEAK [SENSe:]DETector[:FUNCtion] POSitive
#DETECTOR MIN PEAK [SENSe:]DETector[:FUNCtion] NEGative
#DETECTOR SAMPLE [SENSe:]DETector[:FUNCtion] SAMPle
#DETECTOR RMS [SENSe:]DETector[:FUNCtion] RMS
#DETECTOR AVERAGE [SENSe:]DETector[:FUNCtion] AVERage
#DETECTOR QPK [SENSe:]DETector[:FUNCtion] QPEak
#}

#-------------------------------methods to set values to analyzer

    def _SaGetStartFrq(self,numChan=1):        
        """get startfrequency in Hz"""
        inStr=self.GPIBquery('STAR?;')
        if inStr is None:
            inStr=self._startFrq
        return np.float(inStr)
        
    def _SaGet_nPoints(self):
        """get number of Points to be measured"""
        inStr=self.GPIBquery('BAND:RESOLUTION?;')
        return np.int(inStr)        
        
    def _SaGet_Span(self):
        """get span (frequency range)"""
        inStr=self.GPIBquery('SENS:FREQUENCY:SPAN?;')
        return np.float(inStr)        

    def _SaGet_CenterFrequency(self):
        """get central frequency"""
        inStr=self.GPIBquery('SENS:FREQUENCY:CENTER?;')
        return np.float(inStr)        

    def _SaGet_StartFrequency(self):
        """get start frequency"""
        inStr=self.GPIBquery('SENS:FREQUENCY:START?;')
        return np.float(inStr)        

    def _SaGet_StopFrequency(self):
        """get stop frequency"""
        inStr=self.GPIBquery('SENS:FREQUENCY:STOP?;')
        return np.float(inStr)        

    def _SaGet_ResolutionBandWidth(self):
        """get Resolution bandwidth in Hz"""
        #[SENSe<1|2>:]BANDwidth|BWIDth:PLL AUTO | HIGH | MEDium | LOW
        #SENS:BWID:FFT:MODE WIDE | AUTO | NARR
        inStr=self.GPIBquery('BAND:RESOLUTION?;')
        return np.float(inStr)        

    def _SaGet_VideoBandWidth(self):
        #[SENSe<1|2>:]BANDwidth|BWIDth:VIDeo 1Hz...10MHz
        #[SENSe<1|2>:]BANDwidth|BWIDth:VIDeo:AUTO ON | OFF
        #[SENSe<1|2>:]BANDwidth|BWIDth:VIDeo:RATio 0.01...1000
        #[SENSe<1|2>:]BANDwidth|BWIDth:VIDeo:TYPE LINear | LOGarithmic
        if self._SaGet_ResolutionBandWidthType() == 'NORM':
            inStr=self.GPIBquery('BAND:VID?;')
            return np.float(inStr)
        else :
            return -1
            
    def _SaGet_ResolutionBandWidthType(self):
        """get resolution bandwidth type"""
        #{[SENSe<1|2>:]BANDwidth|BWIDth[:RESolution]:TYPE NORMal | FFT | CFILter | RRC | P5
        inStr=self.GPIBquery('BAND:RES:TYPE?;')
        return np.str(inStr)
        #IeeeBus.Enter( FAnt, 100, FLen, IEC_adr, FStat);

    def _SaGet_VideoBandWidthType(self):
        """get video bandwidth type"""
        if self._SaGet_ResolutionBandWidthType() == 'NORM':
            inStr=self.GPIBquery('BAND:VID:TYPE?;')
            return np.str(inStr)
        else :
            return 'None'

# ---------------------------------------------------------------------------

    def _Sa_ReceiveDataNow(self):
        StartFrq=self._SaGet_StartFrequency();
        StopFrq=self._SaGet_StopFrequency();
        nPoints=self._SaGet_nPoints()
        ResBandWidth=self._SaGet_ResolutionBandWidth()
        VidBandWidth=self._SaGet_VideoBandWidth()
        RBWType=self._SaGet_ResolutionBandWidthType()
        # IeeeBus.Send(IEC_Adr, 'SWE:COUN?', stat);  // Number of sweeps (average)
        #  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr, FStat);
        #DISP:WIND:TRAC:Y:RLEV -60dBm
        #DISP:WIND:TRAC:Y:SPAC LOG
        #DISP:WIND:TRAC:Y 100DB}        
        #{
        #  IeeeBus.Send(IEC_adr, 'BAND:PLL?;', FStat); // RBW=1kHz
        #  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr, FStat);
        #  IeeeBus.Send(IEC_adr, 'INPUT:ATT?;', FStat); // RBW=1kHz
        #  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr, FStat);
        #  IeeeBus.Send(IEC_adr, 'DISP:WIND:TRAC:Y:RLEV?;', FStat); // RBW=1kHz
        #  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr, FStat);
        #  }
        #IeeeBus.Send(IEC_adr, 'FORM REAL,32', stat);  // Set the ouput format to a binary format */
        #IeeeBus.Send(IEC_adr, 'TRACE:DATA? TRACE1', stat);  //
        self._Sa_ReceiveData()

    def _Sa_ReceiveData(self):
        self.GPIBwrite('FORM REAL,32') # Set the ouput format to a binary format
        dataBuffer = self.GPIBqueryBin('TRACE:DATA? TRACE1')
        # Buffer starts with # then number of Bytes to encode the total number of elements
        # then total number of elements 
        # then data
        char1=np.frombuffer(dataBuffer,dtype='a1',offset=0,count=1)
        if char1[0]<>'#':
            print 'Error'
        numBytesStr=np.frombuffer(dataBuffer,dtype='a1',offset=1,count=1)
        numBytes=np.int(numBytesStr[0])
        tmpValStr=np.frombuffer(dataBuffer,dtype='S'+str(numBytes),offset=2,count=1)
        ByteSizeOfData=np.int(tmpValStr[0])
        nDataPoints=ByteSizeOfData/4
        data=np.frombuffer(dataBuffer,dtype='float32',offset=2+numBytes, count=nDataPoints)
        finalCharacter=np.frombuffer(dataBuffer,dtype='a1',offset=2+numBytes+ByteSizeOfData,count=1)
        if finalCharacter[0] <>'\n':
            print 'error'
        return data

#-----------------------------------------------------------------------------

    def _Sa_Measure(self):
        self.setToAnalyzer()
        tmpStr="{:6.3f}".format(self.Attenuator)
        outstr=':INPut:ATTenuation '+ tmpStr
        self.GPIBwrite(outstr)
        tmpStr="{:6.3f}".format(self.Reference)
        outstr='DISP:WIND:TRAC:Y:RLEV '+ tmpStr+'dBm'
        self.GPIBwrite(outstr)
        # Setup display of Analyzer
        self.GPIBwrite('DISP:WIND:TRAC:Y:SPAC LOG')
        self.GPIBwrite('DISP:WIND:TRAC:Y 150dB')
        #initialize measurement
        self.GPIBwrite('INIT:CONT OFF;:INIT')
        self.GPIBwrite('*WAI')
        self.data=self._Sa_ReceiveData()
        return True

#-----------------------------------------------------------------------------
    def setToInstr_centerFrq(self, centerfrq=None):
        if centerfrq is None:
            centerfrq=self.getCenterfrq()
        else :
            self.setCenterSpan(inCenterFreq=centerfrq, inSpan=self.getSpan())
        tmpStr="%10.3f" % centerfrq
        outstr= ':SENS:FREQUENCY:CENTER '+tmpStr+'GHz;'
        self.GPIBwrite(outstr )

    def setToInstr_spanFrq(self, spanfrq=None):
        if spanfrq is None:
            spanfrq=self.getSpan()
        else :
            self.setCenterSpan(inCenterFreq=self.getCenterfrq(), inSpan=spanfrq)
        tmpStr="%6.3f" % spanfrq
        outstr=':SENS:FREQUENCY:SPAN '+ tmpStr +'MHz;'
        self.GPIBwrite(outstr ) # Set analyzer span to ??? MHz

    def getFromInstr_nPoints(self):
        result_str=self.GPIBquery('SWE:POIN?;')
        if result_str <> '':
            result=int(result_str)
            self.nPoints=result
        return self.nPoints

    def setToInstr_nPoints(self, n_points=None):
        if n_points is None:
            n_points=self.nPoints
        else :
            self.nPoints=n_points
        outstr='SWE:POIN '+str(self.nPoints);
        self.GPIBwrite(outstr);

    def setToInstr_resBandWidth(self, band_width=None):
        if band_width is None:
            band_width=self.resBandWidth
        else :
            self.resBandWidth=band_width
        tmpStr="%10.3f" % self.resBandWidth
        outstr='BAND:RES '+tmpStr+'Hz;' # RBW=1kHz
        self.GPIBwrite(outstr);

    def setToInstr_vidBandWidth(self, band_width=None):
        if band_width is None:
            band_width=self.vidBandWidth
        else :
            self.vidBandWidth=band_width
        tmpStr="%10.3f" % self.vidBandWidth
        outstr='BAND:VID '+tmpStr+'Hz;' # VBW=1kHz
        self.GPIBwrite(outstr);

    def setToInstr_attenuation(self, attenuation=None):
        if attenuation is None:
            attenuation=self.attenuation
        else :
            self.attenuation=attenuation        
        tmpStr="%6.3f" % self.attenuation
        outstr=':INPut:ATTenuation '+ tmpStr
        self.GPIBwrite(outstr )

    def setToInstr_reference(self, reference=None):
        if reference is None:
            reference=self.reference
        else :
            self.reference=reference        
        tmpStr="%6.3f" % self.reference
        outstr='DISP:WIND:TRAC:Y:RLEV '+ tmpStr+'dBm';
        self.GPIBwrite(outstr);

    def setDetector(self, detector=None):
        if detector is None:
            detector=self.detector
        else:
            self.detector=detector            
        self.GPIBwrite('SENSE:DETECTOR:FUNCTION '+detector) # Set Detector

#
#function TSA_FSU26.GetFromInstr_Span : double;
#begin
        #self.GPIBwrite('SENS:FREQUENCY:SPAN?;');
#  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr);
#  Result=StrToFloat(FAnt);
#end;
#
#function TSA_FSU26.GetFromInstr_CenterFrequency : double;
#begin
        #self.GPIBwrite('SENS:FREQUENCY:CENTER?;');
#  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr);
#  if FAnt <> '' then Result=StrToFloat(FAnt)
#  else Result = GetCenterFreq();
#end;
#
#function TSA_FSU26.GetFromInstr_StartFrequency : double;
#begin
        #self.GPIBwrite('SENS:FREQUENCY:START?;');
#  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr);
#  if FAnt <> '' then Result=StrToFloat(FAnt)
#  else Result = StartFreq*1e9;
#end;
#
#function TSA_FSU26.GetFromInstr_StopFrequency : double;
#begin
        #self.GPIBwrite('SENS:FREQUENCY:STOP?;');
#  IeeeBus.Enter( FAnt, 100, FLen, IEC_adr);
#  if FAnt <> '' then Result=StrToFloat(FAnt)
#  else Result = StopFreq*1e9;
#end;
#
    def GetFromInstr_ResolutionBandWidth(self):
        result=self.GPIBquery('BAND:RESOLUTION?;'); # Resolution Bandwidth in Hz!!!
        if result <> '':
            result=float(result)
        else:  
            result = self.resBandWidth*1e3;
        return result

    def doMeasure(self):
        self.setToInstr_centerFrq()
        #[SENSe<1|2>:]BANDwidth|BWIDth[:RESolution]:TYPE NORMal | FFT | CFILter | RRC | P5
        if self.RBWType <> 'unknown':
            outstr=':SENS:BAND:RES:TYPE '+ self.RBWType
        else: 
            outstr=':SENS:BAND:RES:TYPE NORM'
        self.GPIBwrite(outstr);
        self.setToInstr_resBandWidth()
        self.setToInstr_vidBandWidth()
        self.setToInstr_spanFrq()
        self.setToInstr_attenuation()
        self.setToInstr_reference()
        self.GPIBwrite('DISP:WIND:TRAC:Y:SPAC LOG')
        self.GPIBwrite('DISP:WIND:TRAC:Y 150dB')
        self.setToInstr_nPoints()
        self.GPIBwrite('INIT:CONT OFF;:INIT') # Set the ouput format to a binary format
        self.GPIBwrite('*WAI') # Wait for measurement to be finished
        self.GPIBwrite('FORM REAL,32') # Set the ouput format to a binary format
        self.GPIBwrite('TRACE:DATA? TRACE1') #
        self.receiveData()

    def receiveData(self):
        return self._Sa_ReceiveData()

    def _SaSet_StartFrequency(self, inValue=None):
        """Set startfrequency in GHz to analyzer"""
        if inValue is None: 
            inValue=self.startFrq
        tmpStr="{:9.6f}".format(inValue)
        outstr= ':SENS:FREQUENCY:START '+tmpStr+'GHz;'
        #import pdb; pdb.set_trace()
        self.GPIBwrite(outstr)

    def _SaSet_StopFrequency(self, inValue=None):
        """Set stopfrequency in GHz to analyzer"""
        if inValue is None: 
            inValue=self.stopFrq
        tmpStr="{:9.6f}".format(inValue)
        outstr= ':SENS:FREQUENCY:STOP '+tmpStr+'GHz;'
        #import pdb; pdb.set_trace()
        self.GPIBwrite(outstr)

    def _SaSet_nPoints(self, inValue=None):
        """Set number iof Points for the measurement to analyzer"""
        if inValue is None: 
            inValue=self.nPoints
        tmpStr="{:d}".format(self.nPoints)
        outstr="SWE:POIN "+tmpStr
        self.GPIBwrite(outstr)
        
    def setToAnalyzer(self, continuousFlag=False):
        """Set all values from Object to the spectrum analyzer
        if continuousFlag is True: the analyzer is set into continuous sweep mode"""
        self._SaSet_StartFrequency()
        self._SaSet_StopFrequency()
        self._SaSet_nPoints()
        self.setDetector(self.detector);
        self.setToInstr_resBandWidth()
        self.setToInstr_vidBandWidth()
        tmpStr="%6.3f" % self.span
        outstr=':SENS:FREQUENCY:SPAN '+ tmpStr +'MHz;'
        self.GPIBwrite(outstr ) # Set analyzer span to ??? MHz
        self.setToInstr_nPoints()
        self.GPIBwrite('INIT:CONT 1'); # set the analyzer to continuous mode for manual use

    def getFromAnalyzer(self):
        self.detector=self.setDetector(self.detector);
        tmpStr="%9.6f" % self.centerFreq
        outstr= ':SENS:FREQUENCY:CENTER '+tmpStr+'GHz;'
        self.GPIBwrite(outstr ) #Instrument Preset
        self.setToInstr_resBandWidth()
        self.setToInstr_vidBandWidth()
        tmpStr="%6.3f" % self.span
        outstr=':SENS:FREQUENCY:SPAN '+ tmpStr +'MHz;'
        self.GPIBwrite(outstr ) # Set analyzer span to ??? MHz
        self.setToInstr_nPoints()
        self.GPIBwrite('INIT:CONT 1'); # set the analyzer to continuous mode for manual use

    def setDataFormatBin32(self):
        self.GPIBwrite(':FORM REAL, 32')
        self.GPIBwrite(':FORM:BORD:NORM') # least significant bit is transferred first (little endian). 
        #self.GPIBwrite(':FORM:BORD:SWAP') # The most significant bit is transferred first (big endian).
        self.GPIBwrite('SYST:COMM:GPIB:RTER EOI')

    def getBinary32(self,numTrac=1,numChan=1):
#Block Data Format
#Block Data is a transmission format which is suitable for the transmission of large amounts of Data. A
#command using a block Data parameter with definite length has the following structure:
#Example: HEADer:HEADer #45168xxxxxxxx
#The hash symbol # introduces the Data block. The next number indicates how many of the following digits
#describe the length of the Data block. In the example the 4 following digits indicate the length to be 5168
#bytes. The Data bytes follow. During the transmission of these Data bytes all End or other control signs are
#ignored until all bytes are transmitted.
#A #0 combination introduces a Data block of indefinite length. The use of the indefinite format requires a
#NL^END message to terminate the Data block. This format is useful when the length of the transmission is
#not known or if speed or other considerations prevent segmentation of the Data into blocks of definite
#length.        
#  IeeeBus.Send(IEC_adr, 'FORM REAL,32', stat);  // Set the ouput format to a binary format */
#  IeeeBus.Send(IEC_adr, 'TRACE:DATA? TRACE1', stat);  //
        self.setDataFormatBin32()
        dataRaw=self.GPIBqueryBin('TRACE:DATA? TRACE1')
        dt = np.dtype('float32')
        numOfPoints=self.nPoints
        #numOfPoints=self.getFromInstr_nPoints()
        if dataRaw is None:
            #Only simulations
            print('Simulated Data')
            DataCom=np.zeros(numOfPoints, dtype=np.float32)
            return DataCom
        if dataRaw[0] <> '#':
            print('DataRaw is not OK')
        endIndex= 1+long(dataRaw[1])
        numOfBytesTransfered=long(dataRaw[2:endIndex+1])
        startIndex=endIndex+1
        if numOfPoints*4 <> numOfBytesTransfered:
            print('not correctly transfered the Data!!!')
            print('Transfered binary data: ', numOfBytesTransfered, numOfPoints*4)
        self.data=np.frombuffer(dataRaw, dtype=dt, count=numOfPoints, offset=startIndex)
        return self.data

# --------------- File handling -----------------------------
    def xmlSaveData(self, f):
        import mesopylib.load_save.xmlOwn as xmlOwn
        f.write('<SprectrumAnalyzerData>\n')        
        data=self.data
        xmlOwn.xmlSaveBinaryArray(f, data, name='Spectra')
        f.write('</SprectrumAnalyzerData>\n')

    def xmlSaveHeader(self, f):
        from mesopylib.load_save.xmlOwn import xmlCreateKeyValue
        import xml.dom.minidom as dom
        base_tag = dom.Element('SpectrumAnalyzerHeader') 
        base_tag.setAttribute('Version', '0.1')
#        base_tag.setAttribute('Name', self.vnaName)
#        base_tag.setAttribute('Model', self.vnaModelStr)
#        base_tag.setAttribute('Softwareversion', self.vnaSoftwareVersionStr)
        f_tag=dom.Element('Frequency') 
        xmlCreateKeyValue('Startfrequency','{:E}'.format(np.float(self.startFrq)), attr=[('unit','Hz')], base_tag=f_tag)
        xmlCreateKeyValue('Stopfrequency', '{:E}'.format(np.float(self.stopFrq)), attr=[('unit','Hz')], base_tag=f_tag)
#        xmlCreateKeyValue('Stepfrequency', '{:E}'.format(np.float(self.stepFrq)), attr=[('unit','Hz')], base_tag=f_tag)
        base_tag.appendChild(f_tag)
        prop_tag=dom.Element('Properties') 
#        xmlCreateKeyValue('Statusfilename', self._statefilename, base_tag=prop_tag)
#        xmlCreateKeyValue('NumberOfPoints', self.numberOfTotalPoints, base_tag=prop_tag)
#        xmlCreateKeyValue('Averaging', self.averageFlag, base_tag=prop_tag)
#        xmlCreateKeyValue('AverageNumber', self._averageNum, base_tag=prop_tag)
#        xmlCreateKeyValue('Power', self.power, attr=[('unit','dB')], base_tag=prop_tag)
#        if self.sweepTime is not None:
#            xmlCreateKeyValue('Sweeptime', self.sweepTime, attr=[('unit','s')], base_tag=prop_tag)
        xmlCreateKeyValue('ResBandwidth', self.resBandWidth, attr=[('unit','Hz')], base_tag=prop_tag)
        xmlCreateKeyValue('VidBandwidth', self.vidBandWidth, attr=[('unit','Hz')], base_tag=prop_tag)
        if self.startTime is not None:
            xmlCreateKeyValue('StartofMeasurement', self.startTime, base_tag=prop_tag)
        if self.endTime is not None:
            xmlCreateKeyValue('EndofMeasurement', self.endTime, base_tag=prop_tag)
#        outStr= self.fileName
#        xmlCreateKeyValue('Filename', outStr, base_tag=prop_tag)
#        outStr= self.info
#        xmlCreateKeyValue('Info', outStr, base_tag=prop_tag)
        base_tag.appendChild(prop_tag)
        base_tag.writexml(f, "", "\t", "\n")

    def xmlSave(self, filename=None, f=None, fileOverwrite=False, standard=None, cal_kit=None):
        open_flag=(f==None)
        if open_flag:
            if filename is None:
                filename=saveDialog(filename=filename, name=None, path=None, wildcard='*.xmldat', title='Save xml data file')
#            while os.path.isfile(filename) and not fileOverwrite:
#                reply=overwriteDialog(fileName=filename)
#                if reply==0:
#                    filename=saveDialog(filename=filename, name=None, path=None, wildcard='*.xmldat', title='Save xml data file')
#                    if filename=='':
#                        reply=1
#                        return False
#                    else:
#                        #filename=filename[0]
#                        fileOverwrite=True
#                if reply==2: # Cancel pressed
#                    print('No file saved!!!')
#                    return False
#                if reply==1: # Cancel pressed
#                    fileOverwrite=True
            # create first line!
            self.fileName=filename
            f = open(filename, 'wb')
            f.write('<?xml version="1.0" ?>\n')
            f.write('<MesoNiceXMLFile Version="0.1">\n')
        txt='<SpectrumAnalyzer>\n'
        f.write(txt)
        self.xmlSaveHeader(f)
        self.xmlSaveData(f)
        f.write('</SpectrumAnalyzer>\n')
        if open_flag:
            f.write('</MesoNiceXMLFile>')
            f.close()
        return True
        
if __name__ == "__main__":
    #gpib=GPIB_Base()
    sa=SA_FSU26(GPIBAddr=20, GPIBBoard=0)
    sa.init()
    
    pass
