#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:02:32 2017

@author: jacoblashner
"""

import Telescope as tp
import thermo as th
import numpy as np
import matplotlib.pyplot as plt

pW = 10 ** 12
GHz = 10**9

expDir = "../Experiments/small_aperture/LargeTelescope/"    
#expDir =  "../Experiments/small_aperture/WarmHWP/"
atmFile = "../src/Atacama_1000um_60deg.txt"

theta = 0
#hwpFile = "../HWP_Mueller/Mueller_AR/Mueller_V2_nu150.0_no3p068_ne3p402_ARcoat_thetain%d.0.txt"%theta
hwpFile = "../HWP_Mueller/AHWP_Mueller_90to150GHz_thetainc0deg.txt"

bid = 2

opts = {'theta': np.deg2rad(theta)}

tel = tp.Telescope(expDir, atmFile, hwpFile, bid, **opts)



hwp = tel.elements[tel.hwpIndex]
print th.powFromSpec(tel.freqs, hwp.unpolIncident)*pW


print "a2:", tel.a2 * 100
print "A2 (KRJ)", tel.A2 /tel.cumEff(0, tel.det.band_center)  * pW
print "A4 (KRJ)", tel.A4 /tel.cumEff(0, tel.det.band_center)  /th.kB / (tel.det.band_center * tel.det.fbw)
print "Tel Efficiency", tel.cumEff(0, tel.det.band_center)



print tel.displayTable()