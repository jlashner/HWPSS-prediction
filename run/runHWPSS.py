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
import json
import simplejson
from StringIO import StringIO

pW = 10 ** 12
GHz = 10**9

config = json.load( open ("config.json") )  
tel = tp.Telescope(config)


print "a2:", tel.a2 * 100
print "A2 (KRJ)", tel.A2 /tel.cumEff(0, tel.det.band_center)  * pW
print "A4 (KRJ)", tel.A4 /tel.cumEff(0, tel.det.band_center)  /th.kB / (tel.det.band_center * tel.det.fbw)
print "Tel Efficiency", tel.cumEff(0, tel.det.band_center)

