# -*- coding: utf-8 -*-

from VNA_single_meas import vna_single_meas
import numpy as np
import mesopylib.instrument_control.MotorPDS01 as psd
import time

ps = psd.MotorPDS01()

#Motor 4 is used in setup
ps.openPort()
ps.findOrigin(4)
time.sleep(30)

savePath = "./measurements/21042021/"
fileNamePattern = "non_wely_motor_pos_{}.xmldat"

motor_pos_array = np.linspace(0, 2e5, 401, dtype = int)

for pos in motor_pos_array:
    fileName = fileNamePattern.format(pos)
    ps.moveToAbs(4, pos)
    time.sleep(1)
    vna_single_meas(savePath, fileName)